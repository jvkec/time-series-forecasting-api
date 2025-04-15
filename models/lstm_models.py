import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from models.base_model import BaseForecastModel
from sklearn.preprocessing import MinMaxScaler

class LSTMNetwork(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class LSTMModel(BaseForecastModel):
    def __init__(self, n_lags=7, n_epochs=300, lr=0.01):
        self.n_lags = n_lags
        self.n_epochs = n_epochs
        self.lr = lr
        self.scaler = MinMaxScaler()
        self.model = LSTMNetwork()
        self.last_seq = None

    def _prepare_data(self, series):
        X, y = [], []
        for i in range(len(series) - self.n_lags):
            X.append(series[i:i+self.n_lags])
            y.append(series[i+self.n_lags])
        X = np.array(X)
        y = np.array(y)
        return torch.tensor(X).float().unsqueeze(-1), torch.tensor(y).float().unsqueeze(-1)

    def fit(self, series: pd.Series):
        values = series.values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values).flatten()

        X, y = self._prepare_data(scaled_values)
        X = X.unsqueeze(-1)

        self.last_seq = torch.tensor(scaled_values[-self.n_lags:]).float().reshape(1, self.n_lags, 1)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        for _ in range(self.n_epochs):
            output = self.model(X)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def predict(self, steps: int) -> pd.Series:
        self.model.eval()
        seq = self.last_seq.clone()
        preds = []

        with torch.no_grad():
            for _ in range(steps):
                pred = self.model(seq).item()
                preds.append(pred)
                next_input = torch.tensor([[pred]]).float().unsqueeze(0)
                seq = torch.cat((seq[:, 1:], next_input), dim=1)

        # Inverse transform predictions to original scale
        preds = np.array(preds).reshape(-1, 1)
        preds = self.scaler.inverse_transform(preds).flatten()
        return pd.Series(preds)

    def predict_with_ci(self, steps: int):
        preds = self.predict(steps)
        std = np.std(preds) * 0.1  # very rough CI estimate
        return preds, preds - std, preds + std
