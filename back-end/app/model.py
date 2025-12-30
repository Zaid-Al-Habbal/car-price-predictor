import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "rf_model_pipeline_v1.pkl"


def normalize_mileage(df: pd.DataFrame) -> pd.Series:
    df = df.copy()

    df["mileage_unit"] = df["mileage"].str.split().str[-1]
    df["mileage"] = df["mileage"].astype(str).str.extract(r"([\d\.]+)", expand=False)
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")

    mask_petrol = (df["mileage_unit"] == "km/kg") & (df["fuel"] == "Petrol")
    mask_diesel = (df["mileage_unit"] == "km/kg") & (df["fuel"] == "Diesel")
    mask_cng = (df["mileage_unit"] == "km/kg") & (df["fuel"] == "CNG")
    mask_lpg = (df["mileage_unit"] == "km/kg") & (df["fuel"] == "LPG")

    df.loc[mask_petrol, "mileage"] /= 0.74
    df.loc[mask_diesel, "mileage"] /= 0.832
    df.loc[mask_lpg, "mileage"] /= 0.54
    df.loc[mask_cng, "mileage"] /= 0.128

    return df["mileage"]


def group_seats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print(f"group seats df: {df}")

    conditions = [
        df["seats"] < 5,
        df["seats"] == 5,
        df["seats"] > 5,
    ]
    choices = ["less_than_five", "five", "more_than_five"]
    df["seats"] = np.select(conditions, choices, default="missing")
    df["seats"] = df["seats"].astype("category")

    return df


def group_rare_names(df: pd.DataFrame, threshold: int = 10) -> pd.Series:
    df = df.copy()

    name_counts = df["name"].value_counts()
    rare_names = name_counts[name_counts < threshold].index
    df["name"] = df["name"].replace(rare_names, "other")

    return df["name"]


def group_rare_fuel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fuel"] = df["fuel"].replace({"CNG": "other", "LPG": "other"})
    return df


def update_owner_grouping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["owner"] = df["owner"].replace(
        {
            "Third Owner": "Third & Above Owner",
            "Fourth & Above Owner": "Third & Above Owner",
            "Test Drive Car": "First Owner",
        }
    )
    return df


def convert_year_to_age(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["age"] = 2026 - df["year"]
    return df["age"]


class BaseNumericFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["engine"] = pd.to_numeric(X["engine"].str.split().str[0], errors="coerce")
        X["max_power"] = pd.to_numeric(
            X["max_power"].str.split().str[0], errors="coerce"
        )

        X["mileage"] = normalize_mileage(X)
        X["age"] = convert_year_to_age(X)

        X.drop(columns=["year", "fuel"], inplace=True)

        return X

    def get_feature_names_out(self, input_features=None):
        return ["engine", "max_power", "mileage", "age", "km_driven"]


class InteractionFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        eps = 1e-6
        X = X.copy()

        X["engine_mileage_ratio"] = X["engine"] / (X["mileage"] + eps)
        X["km_driven_age_interaction"] = X["km_driven"] * X["age"]

        return X

    def get_feature_names_out(self, input_features=None):
        return [
            "engine_mileage_ratio",
            "km_driven_age_interaction",
            "engine",
            "max_power",
            "mileage",
            "age",
            "km_driven",
        ]


class NameTranformation(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: int = 5):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["name"] = X["name"].str.split().str[0]
        X["name"] = group_rare_names(X, self.threshold)

        X.rename(columns={"name": "brand_effect"}, inplace=True)

        return X

    def get_feature_names_out(self, input_features=None):
        return ["brand_effect"]


def _register_pickle_compat_symbols() -> None:
    symbols = {
        "normalize_mileage": normalize_mileage,
        "group_seats": group_seats,
        "group_rare_names": group_rare_names,
        "group_rare_fuel": group_rare_fuel,
        "update_owner_grouping": update_owner_grouping,
        "convert_year_to_age": convert_year_to_age,
        "BaseNumericFeatures": BaseNumericFeatures,
        "InteractionFeatures": InteractionFeatures,
        "NameTranformation": NameTranformation,
    }

    for module_name in ("__main__", "__mp_main__"):
        module = sys.modules.get(module_name)
        if module is None:
            continue

        for name, obj in symbols.items():
            if not hasattr(module, name):
                setattr(module, name, obj)


class CarPriceModel:
    def __init__(self):
        
        _register_pickle_compat_symbols()

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        self.model = joblib.load(MODEL_PATH)
        
        self.feature_names = list(self.model.feature_names_in_)
        

    def predict(self, car_features: pd.DataFrame):
        return self.model.predict(car_features)