from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import numpy as np
import pandas as pd
from crypto_volatility_lab.data_construction.cryptoScraper import CryptoScraper
from crypto_volatility_lab.data_construction.featuresCreator import FeaturesCreator
from crypto_volatility_lab.data_construction.timeSeriesCreator import TimeSeriesCreator
import matplotlib.pyplot as plt
import io
from fastapi.responses import Response



cached_data = None

app = FastAPI(title="Crypto Volatility Lab API")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/scrape", response_class=HTMLResponse)
def scrape_crypto_html(
    request: Request,
    crypto: str = Query(..., description="Nom de la crypto (ex: BTC-USD)")
):
    """Scrape les données et les stocke pour les étapes suivantes."""
    global cached_data  # Utilisation de la variable globale

    try:
        scraper = CryptoScraper()
        data = scraper.get_data_for_currency(crypto)

        if data.empty:
            raise HTTPException(status_code=404, detail=f"Aucune donnée trouvée pour {crypto}")

       
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values(by="Date", ascending=False).reset_index(drop=True)

        
        numeric_columns = ["Open", "High", "Low", "Close", "Adj", "Volume"]
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col].astype(str).str.replace(",", ""), errors="coerce")

        cached_data = data  

        print("📌 Scraped Data After Sorting:")
        print(data.head())  

        return templates.TemplateResponse(
            "scrape.html",
            {"request": request, "crypto": crypto, "data": data.to_dict(orient="records")},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compute_features", response_class=HTMLResponse)
def compute_features_html(
    request: Request,
    crypto: str = Query(..., description="Cryptocurrency ticker (ex: BTC-USD)")
):
    """Génère les features en utilisant les données scrappées."""
    global cached_data  

    try:
        if cached_data is None:
            raise HTTPException(status_code=400, detail="Aucune donnée scrappée disponible. Exécutez `/scrape` d'abord.")

        df = cached_data.copy() 

        print("📌 Features Using Cached Data:")
        print(df.head())

        if "Close" not in df.columns:
            raise HTTPException(status_code=400, detail="Les données ne contiennent pas de prix de clôture.")

        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df.dropna(subset=["Close"], inplace=True)

        
        features_creator = FeaturesCreator(df, log_returns_column_name="Close", volatility_column_name="Close")
        features_creator.create_all_features()
        features = features_creator.transformed_data.dropna()

        
        features["Date"] = pd.to_datetime(features["Date"])
        #features = features.sort_values(by="Date", ascending=False).reset_index(drop=True)

        print("📌 Features Data After Processing:")
        print(features.head())
        print("📌 Colonnes disponibles après feature engineering:")
        print(features.columns)

        cached_data = features.copy()



        return templates.TemplateResponse(
            "features.html",
            {"request": request, "crypto": crypto, "features": features.to_dict(orient="records")},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/plot_features")
def plot_features(feature: str):
    """Génère un graphique en fonction du type de feature demandé."""
    global cached_data
    if cached_data is None:
        raise HTTPException(status_code=400, detail="Aucune donnée disponible. Exécutez `/compute_features` d'abord.")

    df = cached_data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date", ascending=True)

    # 📌 Vérifier quelles colonnes sont disponibles
    print("📌 Colonnes disponibles dans df avant le tracé des graphes:", df.columns)

    plt.figure(figsize=(10, 5))

    if feature == "volatility":
        if "volatility_weekly_smoothed" in df.columns and "volatility_monthly_smoothed" in df.columns:
            plt.plot(df["Date"], df["volatility_weekly_smoothed"], label="Weekly Volatility", color='blue')
            plt.plot(df["Date"], df["volatility_monthly_smoothed"], label="Monthly Volatility", color='red')
        else:
            raise HTTPException(status_code=500, detail="Les colonnes de volatilité ne sont pas disponibles.")


    plt.xlabel("Date")
    plt.ylabel(feature.capitalize())
    plt.title(f"{feature.capitalize()} Graph")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/create_time_series", response_class=HTMLResponse)
def create_time_series_html(
    request: Request,
    crypto: str = Query(..., description="Cryptocurrency ticker (ex: BTC-USD)"),
    window_size: int = Query(21, description="Window size for volatility computation"),
):
    """Génère une série temporelle en utilisant les dernières données scrappées."""
    global cached_data  

    try:
        if cached_data is None:
            raise HTTPException(status_code=400, detail="Aucune donnée scrappée disponible. Exécutez `/scrape` d'abord.")

        df = cached_data.copy()  

        print("📌 Time Series Using Cached Data Before Processing:")
        print(df.head()) 

        
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by="Date", ascending=False).reset_index(drop=True)

        print("📌 Time Series After Sorting:")
        print(df.head())  

        
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0)

        print("📌 Log Returns Computed on Data:")
        print(df[["Date", "Close", "Log_Returns"]].head())  

        
        time_series_creator = TimeSeriesCreator(df.copy(), date_column_name="Date", value_column_name="Close")
        time_series = time_series_creator.create_time_series(window_size)

        print("📌 Final Time Series Sent to Template:")
        print(time_series.head())  

        return templates.TemplateResponse(
            "time_series.html",
            {"request": request, "crypto": crypto, "window_size": window_size, "time_series": time_series.dropna().to_dict(orient="records")},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
