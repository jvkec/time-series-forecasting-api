# Time Series Forecasting API

A Python-based API for automated time series forecasting with model selection, confidence intervals, and anomaly detection.

## Features

- Automatic model selection (ARIMA, Prophet, LSTM)
- Confidence interval support
- Residual-based anomaly detection
- RESTful API via FastAPI
- Modular, extensible architecture

## Tech Stack

- Python 3.11+
- FastAPI
- PyTorch
- Prophet
- Statsmodels
- scikit-learn
- Pandas

## Endpoints

### POST `/forecast`

**Request**
```json
{
  "series": [100, 102, 105, 107, 110, 115]
}
```

**Response**
```json
{
  "model": "LSTMModel",
  "forecast": [116.2, 117.4, 118.9],
  "lower_ci": [114.1, 115.3, 116.8],
  "upper_ci": [118.3, 119.5, 121.0],
  "score": 1.42
}
```

### POST `/detect-anomalies`

**Request**
```json
{
  "series": [100, 101, 102, 200, 104, 105]
}
```

**Response**
```json
{
  "anomalies": ["2024-01-04T00:00:00"],
  "model": "ARIMAModel",
  "forecast": [104.2, 105.1, 106.7],
  "score": 14.3
}
```

## Getting Started

```bash
git clone https://github.com/your-username/time-series-forecasting-api
cd time-series-forecasting-api

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn main:app --reload
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) to explore the Swagger API documentation.

## Future Work

- CSV upload support
- Streamlit dashboard
- Transformer model integration
- Model versioning and persistence