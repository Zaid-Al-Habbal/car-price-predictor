from pydantic import BaseModel
from typing import Literal


class CarFeatures(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: Literal["Diesel", "Petrol", "CNG", "LPG"]
    seller_type: Literal["Individual", "Dealer", "Trustmark Dealer"]
    transmission: Literal["Manual", "Automatic"]
    owner: Literal[
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car",
    ]
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float