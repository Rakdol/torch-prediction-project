import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.train.metrics import RMSE, MAPE, SMAPE, NMAE


def evaluate_model(model, test_loader, nominal=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    y_preds = []
    y_tests = []
    with torch.no_grad():
        for X_test_, y_test_ in test_loader:

            X_test_, y_test_ = X_test_.to(device), y_test_.to(device)
            y_pred = model(X_test_)
            y_tests.append(y_test_.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())
    y_preds = np.concatenate(y_preds, axis=0)
    y_tests = np.concatenate(y_tests, axis=0)

    evaluations = {
        "MAE": mean_absolute_error(y_tests, y_preds),
        "MSE": mean_squared_error(y_tests, y_preds),
        "RMSE": RMSE(y_tests, y_preds),
        "SMAPE": SMAPE(y_tests, y_preds),
        "MAPE": MAPE(y_tests, y_preds),
        "R2": r2_score(y_tests, y_preds),
    }
    if nominal is not None:
        evaluations["NMAE"] = NMAE(y_tests, y_preds, nominal)
    print("Evaluation metrics:")
    for key, value in evaluations.items():
        print(f"{key}: {value:.4f}")

    return evaluations
