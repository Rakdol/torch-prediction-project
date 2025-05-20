import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_log_error


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
