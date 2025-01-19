from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from crypto_volatility_lab.data_construction.cryptoScraper import CryptoScraper

app = FastAPI(title="Crypto Volatility Lab API")


templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/scrape")
def scrape_crypto(crypto: str = Query(..., description="Nom de la crypto (ex: BTC-USD)")):
    try:
        scraper = CryptoScraper()
        data = scraper.get_data_for_currency(crypto)

        if data.empty:
            raise HTTPException(status_code=404, detail=f"Aucune donnée trouvée pour {crypto}")

        return {"crypto": crypto, "data": data.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
