from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from crypto_volatility_lab.data_construction import cryptoScraper

app = FastAPI(title="Crypto Volatility Lab API")

try:
    model = joblib.load(".....")
except:
    model = None

@app.get("/")
def index():
    return {"message": "Bienvenue sur l'API Crypto Volatility Lab"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/scrape")
def scrape_crypto(crypto: str = Query(..., description="Nom de la crypto")):
    try:
        data = cryptoScraper(crypto)  
        return {"crypto": crypto, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PredictionInput(BaseModel):
    crypto: str
    features: list[float]

@app.post("/predict")
def predict_price(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return {"crypto": data.crypto, "predicted_price": prediction}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

