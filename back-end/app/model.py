import joblib
from pathlib import Path
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "rf_model_pipeline_v1.pkl"


class CarPriceModel:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
    
    def predict(self, car_features: pd.DataFrame):
        return self.model.predict(car_features)