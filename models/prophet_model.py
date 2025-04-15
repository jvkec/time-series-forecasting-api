from models.base_model import BaseForecastModel
from prophet import Prophet
import pandas as pd

class ProphetModel(BaseForecastModel):
    def __init__(self):
        self.model = Prophet()
        self.fitted = False

    def fit(self, series: pd.Series):
        df = series.reset_index()
        df.columns = ['ds', 'y']  # Prophet requires 'ds' (datetime) and 'y' (value)
        self.model.fit(df)
        self.fitted = True

    def predict(self, steps: int) -> pd.Series:
        if not self.fitted:
            raise Exception("Model not fitted.")
        
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        return forecast['yhat'][-steps:].reset_index(drop=True)

    def predict_with_ci(self, steps: int) -> tuple:
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        return (
            forecast['yhat'][-steps:].reset_index(drop=True),
            forecast['yhat_lower'][-steps:].reset_index(drop=True),
            forecast['yhat_upper'][-steps:].reset_index(drop=True),
        )
