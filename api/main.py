from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from crypto_volatility_lab.data_construction.cryptoScraper import CryptoScraper
from crypto_volatility_lab.data_construction.featuresCreator import FeaturesCreator
from crypto_volatility_lab.data_construction.timeSeriesCreator import TimeSeriesCreator


app = FastAPI(title="Crypto Volatility Lab API")
templates = Jinja2Templates(directory="templates")


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
        time_series_data[crypto] = (
            time_series_creator.create_time_series(window_size)
            .dropna()
            .to_dict(orient="records")
        )

    time_series_cached_data = time_series_data
    return templates.TemplateResponse(
        "time_series.html",
        {"request": request, "window_size": window_size, "data": time_series_data},
    )


@app.get("/compute_features", response_class=HTMLResponse)
def compute_features_html(request: Request):
    global features_cached_data

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

    return templates.TemplateResponse(
        "features.html",
        {"request": request, "data": features_data},
    )


@app.get("/plot_features")
def plot_features(crypto: str):
    """Génère un graphique spécifique pour une crypto donnée."""
    global features_cached_data

    if crypto not in features_cached_data:
        raise HTTPException(
            status_code=400,
            detail=f"Aucune donnée disponible pour {crypto}. Exécutez `/compute_features` d'abord.",
        )

    df = pd.DataFrame(features_cached_data[crypto])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    if (
        "volatility_weekly_smoothed" in df.columns
        and "volatility_monthly_smoothed" in df.columns
    ):
        ax.plot(
            df["Date"],
            df["volatility_weekly_smoothed"],
            label="Weekly Volatility",
            linestyle="dashed",
            color="blue",
        )
        ax.plot(
            df["Date"],
            df["volatility_monthly_smoothed"],
            label="Monthly Volatility",
            color="red",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.set_title(f"Volatility Graph - {crypto}")
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
