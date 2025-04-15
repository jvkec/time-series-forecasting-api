from abc import ABC, abstractmethod
import pandas as pd

class BaseForecastModel(ABC):
    @abstractmethod
    def fit(self, series: pd.Series):
        pass

    @abstractmethod
    def predict(self, steps: int) -> pd.Series:
        pass

    def predict_with_ci(self, steps: int) -> tuple:
        """Optional: prediction + confidence interval"""
        raise NotImplementedError