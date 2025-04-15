import pandas as pd

def detect_anomalies(series: pd.Series, forecast: pd.Series, threshold: float = 2.5):
    residuals = series[-len(forecast):] - forecast
    std = residuals.std()

    anomalies = (residuals.abs() > threshold * std)
    return anomalies[anomalies].index.tolist()  # return time indices of anomalies
