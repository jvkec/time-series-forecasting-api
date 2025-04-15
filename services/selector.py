from typing import List
import pandas as pd
from sklearn.metrics import mean_absolute_error

class ModelSelector:
    def __init__(self, models: List):
        self.models = models

    def select_and_forecast(self, series: pd.Series, forecast_steps: int):
        best_model = None
        best_score = float("inf")
        best_forecast = None

        # Holdout strategy: last N points for validation
        train_series = series[:-forecast_steps]
        val_series = series[-forecast_steps:]

        for model in self.models:
            try:
                model.fit(train_series)
                forecast = model.predict(forecast_steps)
                score = mean_absolute_error(val_series, forecast)
                if score < best_score:
                    best_score = score
                    best_model = model
                    best_forecast = forecast
            except Exception as e:
                print(f"[WARN] Model {type(model).__name__} failed: {e}")

        if best_forecast is None:
            raise Exception("All models failed to forecast. Try using more data.")

        return {
            "model": type(best_model).__name__,
            "forecast": best_forecast.tolist(),
            "score": best_score
        }
