from fastapi import APIRouter, Request
import pandas as pd

from models.arima_model import ARIMAModel
from models.prophet_model import ProphetModel
from models.lstm_models import LSTMModel
from services.selector import ModelSelector
from services.anomaly import detect_anomalies

models = [
    ARIMAModel(),
    ProphetModel(),
    LSTMModel()
]

router = APIRouter()

@router.post("/forecast")
async def forecast_endpoint(request: Request):
    data = await request.json()
    values = data["series"]
    index = pd.date_range(start="2024-01-01", periods=len(values), freq="D")
    series = pd.Series(values, index=index)

    if len(values) < 15:
        return {"error": "Please provide at least 15 data points."}

    selector = ModelSelector(models)  # uses ARIMA, Prophet, LSTM
    result = selector.select_and_forecast(series, forecast_steps=10)

    return result

@router.post("/detect-anomalies")
async def detect_anomalies_endpoint(request: Request):
    data = await request.json()
    values = data["series"]
    index = pd.date_range(start="2024-01-01", periods=len(values), freq="D")
    series = pd.Series(values, index=index)

    if len(values) < 15:
        return {"error": "Please provide at least 15 data points."}

    selector = ModelSelector(models)
    result = selector.select_and_forecast(series, forecast_steps=10)

    anomalies = detect_anomalies(series, pd.Series(result["forecast"], index=series[-10:].index))

    return {
        "anomalies": anomalies,
        "model": result["model"],
        "forecast": result["forecast"],
        "score": result["score"]
    }

