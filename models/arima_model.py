import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from models.base_model import BaseForecastModel

class ARIMAModel(BaseForecastModel):
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model = None
        self.result = None

    def fit(self, series: pd.Series):
        self.model = ARIMA(series, order=self.order)
        self.result = self.model.fit()

    def predict(self, steps: int) -> pd.Series:
        forecast = self.result.forecast(steps=steps)
        return forecast
    
    def predict_with_ci(self, steps: int) -> tuple:
        forecast_result = self.result.get_forecast(steps=steps)
        mean = forecast_result.predicted_mean
        ci = forecast_result.conf_int()
        lower = ci.iloc[:, 0]
        upper = ci.iloc[:, 1]
        return (mean, lower, upper)
