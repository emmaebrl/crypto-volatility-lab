import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
import numpy as np
from starlette.requests import Request
import pandas as pd
import matplotlib
import pickle

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from crypto_volatility_lab.data_construction.cryptoScraper import CryptoScraper
from crypto_volatility_lab.data_construction.featuresCreator import FeaturesCreator
from crypto_volatility_lab.data_construction.timeSeriesCreator import TimeSeriesCreator
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
import pandas as pd
from typing import Any, Dict, List, Optional
from crypto_volatility_lab.portfolio_optimization.portfolioConstructor import (
    PortfolioConstructor,
)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


app = FastAPI(title="Crypto Volatility Lab API")
templates = Jinja2Templates(directory="templates")
TF_ENABLE_ONEDNN_OPTS = 0


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/scrape", response_class=HTMLResponse)
def scrape_crypto_html(request: Request):
    global scraped_cached_data
    cryptos = ["BTC-USD", "ETH-USD", "LTC-USD"]
    scraped_data = {}

    try:
        scraper = CryptoScraper()
        for crypto in cryptos:
            data = scraper.get_data_for_currency(crypto)
            if data is None or data.empty:
                scraped_data[crypto] = []
            else:
                data["Date"] = pd.to_datetime(data["Date"])
                data = data.sort_values(by="Date", ascending=False).reset_index(
                    drop=True
                )

                numeric_columns = ["Open", "High", "Low", "Close", "Adj", "Volume"]
                for col in numeric_columns:
                    if col in data.columns:
                        data[col] = pd.to_numeric(
                            data[col].astype(str).str.replace(",", ""), errors="coerce"
                        )

                scraped_data[crypto] = data.to_dict(orient="records")

        scraped_cached_data = scraped_data

        return templates.TemplateResponse(
            "scrape.html", {"request": request, "data": scraped_data}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/create_time_series", response_class=HTMLResponse)
def create_time_series_html(request: Request, window_size: int = 21):
    global time_series_cached_data

    if not scraped_cached_data:
        raise HTTPException(
            status_code=400,
            detail="Aucune donnée disponible. Exécutez `/scrape` d'abord.",
        )

    time_series_data = {}

    for crypto, data in scraped_cached_data.items():
        df = pd.DataFrame(data)
        df = df.sort_values(by="Date", ascending=True).reset_index(drop=True)

        time_series_creator = TimeSeriesCreator(
            df, date_column_name="Date", value_column_name="Close"
        )
        df["Log Returns"] = time_series_creator.create_log_return_time_series()
        df["Volatility"] = time_series_creator.create_volatility_time_series(
            window_size
        )
        time_series_data[crypto] = df.dropna().to_dict(orient="records")

    time_series_cached_data = time_series_data
    return templates.TemplateResponse(
        "time_series.html",
        {"request": request, "window_size": window_size, "data": time_series_data},
    )


@app.get("/compute_features", response_class=HTMLResponse)
def compute_features_html(request: Request):
    global features_cached_data
    global features_names

    if not time_series_cached_data:
        raise HTTPException(
            status_code=400,
            detail="Aucune donnée disponible. Exécutez `/scrape` d'abord.",
        )

    features_data = {}

    for crypto, data in time_series_cached_data.items():
        df = pd.DataFrame(data)
        features_creator = FeaturesCreator(
            df,
            log_returns_column_name="Log Returns",
            volatility_column_name="Volatility",
        )
        features_creator.create_all_features()
        features_data[crypto] = features_creator.transformed_data.dropna().to_dict(
            orient="records"
        )

    features_cached_data = features_data
    features_names = features_creator.features_names

    return templates.TemplateResponse(
        "features.html",
        {"request": request, "data": features_data},
    )


def predict_for_model(model_path, data):
    """Predicts the target variable using the given model."""
    pipeline = pickle.load(open(model_path, "rb"))
    return pipeline.predict(data)


@app.get("/predictions", response_class=HTMLResponse)
def predictions_page(request: Request):
    """Render the main prediction page with model type selection."""
    model_types = ["GRU", "LSTM", "LSTMGRU", "TCNN"]
    return templates.TemplateResponse(
        "predictions_menu.html", {"request": request, "model_types": model_types}
    )


@app.get("/predictions/{model_type}", response_class=HTMLResponse)
def predictions_by_model(model_type: str, request: Request):
    """Generate future predictions for all cryptos using the selected model type."""
    global predictions_cached_data
    if not features_cached_data:
        raise HTTPException(
            status_code=400,
            detail="Aucune donnée disponible. Exécutez /compute_features d'abord.",
        )

    predict_data = {}
    for crypto, data in features_cached_data.items():
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["Date"])
        last_date = df["Date"].max()

        # take last date and 30 days before
        df = df[df["Date"] >= last_date - pd.DateOffset(days=29)]
        df = df[features_names]

        model_path = os.path.join("api", "models", crypto, f"{crypto}_{model_type}.pkl")

        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail=f"Modèle {model_type} introuvable pour {crypto}.",
            )

        predictions = predict_for_model(model_path, df)
        print(f"\n✅ Predictions for {crypto} using {model_type}:", predictions)

        predictions = predictions.flatten()
        predict_data[crypto] = [
            {"Date": last_date, "Volatility": df["Volatility"].values[-1]}
        ]

        for i in range(5):
            predict_data[crypto].append(
                {
                    "Date": last_date + pd.DateOffset(days=i + 1),
                    "Volatility": predictions[i],
                }
            )

        print(predict_data[crypto])

    predictions_cached_data = predict_data

    return templates.TemplateResponse(
        "crypto_predictions.html",
        {"request": request, "model_type": model_type, "data": predict_data},
    )


@app.get("/risk_parity", response_class=HTMLResponse)
def risk_parity_page(request: Request, target_vol_factor: Optional[float] = 1.0):
    """
    Génère la page HTML affichant les poids optimisés du portefeuille en utilisant Risk Parity.
    Affiche les deux méthodes : "simple" et "target".
    """

    global predictions_cached_data

    if not predictions_cached_data:
        raise HTTPException(
            status_code=400,
            detail="Aucune prédiction disponible. Exécutez `/predictions/LSTM` d'abord.",
        )

    # 🔹 Sélectionner les valeurs `t+5` de la **deuxième ligne** (index 1)
    lstm_predictions = {
        crypto: data[1]["t+5"] if len(data) > 1 and "t+5" in data[1] else None
        for crypto, data in predictions_cached_data.items()
    }

    # 🔹 Supprimer les cryptos sans valeur valide
    lstm_predictions = {k: v for k, v in lstm_predictions.items() if v is not None}

    if not lstm_predictions:
        raise HTTPException(
            status_code=400,
            detail="Les prédictions `t+5` (ligne 2) ne sont pas disponibles.",
        )

    # 🔹 Création du DataFrame pour le calcul Risk Parity
    volatility_df = pd.DataFrame([lstm_predictions])
    print("\n✅ Volatility DataFrame (t+5, ligne 2) :\n", volatility_df)

    # 🔹 Création de l'instance PortfolioConstructor
    portfolio_constructor = PortfolioConstructor(
        volatility_time_series=volatility_df, target_vol_factor=target_vol_factor
    )

    # 🔹 Calcul des deux méthodes
    optimized_weights = {
        "simple": portfolio_constructor.risk_parity_weights_simple(),
        "target": portfolio_constructor.risk_parity_weights_simple_target(),
    }

    # 🔹 Vérification et conversion des résultats
    latest_weights = {}
    for method, df in optimized_weights.items():
        if isinstance(df, pd.DataFrame):
            latest_weights[method] = df.to_dict(orient="records")
        elif isinstance(df, pd.Series):
            latest_weights[method] = df.to_dict()
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Erreur: `optimized_weights` pour {method} est un float au lieu d'un DataFrame.",
            )

    print("\n✅ Debug: latest_weights après conversion:", latest_weights)

    # 🔹 Envoi des résultats au template
    return templates.TemplateResponse(
        "risk_parity.html",
        {
            "request": request,
            "target_vol_factor": target_vol_factor,
            "data": latest_weights,  # ✅ Contient les poids pour "simple" et "target"
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
