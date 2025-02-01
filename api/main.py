import os
import pandas as pd
import matplotlib
import pickle

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from crypto_volatility_lab.data_construction.cryptoScraper import CryptoScraper
from crypto_volatility_lab.data_construction.featuresCreator import FeaturesCreator
from crypto_volatility_lab.data_construction.timeSeriesCreator import TimeSeriesCreator
from crypto_volatility_lab.portfolio_optimization.portfolioConstructor import (
    PortfolioConstructor,
)

matplotlib.use("Agg")
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
            detail="Aucune donnée disponible. Exécutez /scrape d'abord.",
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
            detail="Aucune donnée disponible. Exécutez /scrape d'abord.",
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
        # df = df[df["Date"] >= last_date - pd.DateOffset(days=32)]
        df = df[features_names]

        model_path = os.path.join("api", "models", crypto, f"{crypto}_{model_type}.pkl")

        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404,
                detail=f"Modèle {model_type} introuvable pour {crypto}.",
            )
        predictions = predict_for_model(model_path, df)

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
def risk_parity_page(request: Request):
    

    global predictions_cached_data

    if not predictions_cached_data:
        raise HTTPException(
            status_code=400,
            detail="Aucune prédiction disponible. Exécutez /predictions/LSTM d'abord.",
        )

    
    all_days_weights = {}

    for day_offset in range(1, 6):  # Pour t+1 à t+5
        day_predictions = {
            crypto: data[day_offset]["Volatility"]
            for crypto, data in predictions_cached_data.items()
            if len(data) > day_offset
        }

        if not day_predictions:
            continue  

        
        volatility_df = pd.DataFrame([day_predictions])
        print(f"\n✅ Volatility DataFrame (t+{day_offset}):\n", volatility_df)

        
        portfolio_constructor = PortfolioConstructor(
            volatility_time_series=volatility_df
        )

        
        weights_df = portfolio_constructor.risk_parity_weights_simple()

        
        all_days_weights[f"t+{day_offset}"] = weights_df.to_dict(orient="records")[0]

    if not all_days_weights:
        raise HTTPException(
            status_code=400,
            detail="Aucune prédiction valide pour les jours t+1 à t+5.",
        )

    print("\n✅ Poids optimisés pour t+1 à t+5:", all_days_weights)

    
    return templates.TemplateResponse(
        "risk_parity.html",
        {
            "request": request,
            "data": all_days_weights,  
        },
    )


if _name_ == "_main_":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)