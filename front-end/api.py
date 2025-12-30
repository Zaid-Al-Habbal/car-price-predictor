import requests

def predict_price(api_url: str, payload: dict) -> float:
    response = requests.post(
        f"{api_url}/predict",
        json=payload,
        timeout=10
    )
    response.raise_for_status()
    return response.json()["predicted_price"]
