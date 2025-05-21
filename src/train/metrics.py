import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_log_error
import torch
import torch.nn as nn


def RMSE(y, y_pred):
    rmse = mean_squared_error(y, y_pred) ** 0.5
    return rmse


def MAPE(y_test, y_pred):
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100


def SMAPE(y_test, y_pred):
    return np.mean((np.abs(y_test - y_pred)) / (np.abs(y_test) + np.abs(y_pred))) * 100


def NMAE(true, pred, nominal):
    absolute_error = np.abs(true - pred)

    absolute_error /= nominal

    target_idx = np.where(true >= nominal * 0.1)

    return 100 * absolute_error[target_idx].mean()


class nMAELoss(nn.Module):
    def __init__(self, nominal):
        super().__init__()
        self.nominal = nominal  # 예: 1500

    def forward(self, pred, true):
        absolute_error = torch.abs(pred - true)
        normalized_error = absolute_error / self.nominal

        # true 값이 nominal * 0.1 이상인 경우만 선택
        mask = true >= (self.nominal * 0.1)
        filtered_error = normalized_error[mask]

        if filtered_error.numel() == 0:
            return torch.tensor(0.0, requires_grad=True).to(pred.device)

        return 100 * filtered_error.mean()
