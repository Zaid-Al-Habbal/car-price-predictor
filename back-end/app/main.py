from fastapi import FastAPI
import pandas as pd

from app.schemas import CarFeatures
from app.model import CarPriceModel


app = FastAPI(
    title="Car Price Predictor API",
    version="1.0.0",
    description="predict car price based on its features using a trained ML pipeline"
)


model = CarPriceModel()


@app.get("/")
def check_health():
    return {"status": "OK"}


@app.post("/predict")
def predict_car_price(features: CarFeatures):
    df = pd.DataFrame([features.model_dump()])

    prediction = model.predict(df)[0]

    return {
        "predicted_price": float(prediction)
    }