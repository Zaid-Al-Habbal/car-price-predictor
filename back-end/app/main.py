from fastapi import FastAPI
import pandas as pd

from .schemas import CarFeatures
from .model import CarPriceModel

##+++++------------%%%%%%%%%%%%%%%%%%%^^^^^^^^^^^^^^$$$$$$$$$$$$$$$$$$$$$$$???????????????????????
from sklearn import set_config


app = FastAPI(
    title="Car Price Predictor API",
    version="1.0.0",
    description="predict car price based on its features using a trained ML pipeline",
)


model = CarPriceModel()


@app.get("/")
def check_health():
    return {"status": "OK"}


@app.post("/predict")
def predict_car_price(features: CarFeatures):
    # BEEEEEEEEEEEEE CAREFULEEEEEE: YOU ARE USEING DATAFRAMES AS OUTPUTS IN YOUR TRANSFORMATIONS
    # IN TRAINING YOUR MODEL SO YOU NEED TO USE THEM FOR INFERENCE ALSO!!!
    set_config(transform_output="pandas")

    df = pd.DataFrame([features.model_dump()])
    prediction = model.predict(df)[0]
    
    return {"predicted_price": float(prediction)}