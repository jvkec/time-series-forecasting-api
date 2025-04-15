from typing import List
import pandas as pd
from sklearn.metrics import mean_absolute_error

class ModelSelector:
    def __init__(self, models: List):
        self.models = models

    def select_and_forecast(self, series: pd.Series, forecast_steps: int):
        best_model = None
        best_score = float("inf")

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
            except Exception as e:
                print(f"[WARN] Model {type(model).__name__} failed: {e}")
        
        if best_model is None:
            raise RuntimeError("All models failed. Try submitting more data or tuning model settings.")

        forecast, lower, upper = best_model.predict_with_ci(forecast_steps)

        return {
            "model": type(best_model).__name__,
            "forecast": forecast.tolist(),
            "lower_ci": lower.tolist(),
            "upper_ci": upper.tolist(),
            "score": best_score
        }