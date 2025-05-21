from src.config.task_configs import (
    LoadDataConfigurations,
    LoadFeatureConfigurations,
    SolarDataConfigurations,
    SolarFeatureConfigurations,
)
from src.dataprep.dataset import LoadDataset, SolarDataset
from src.run.train_runner import run_training
from src.train.predictor import predict
from src.model.base_model import DeepResidualMLP
from src.train.visual import plot_predictions
from src.train.metrics import nMAELoss
import matplotlib.pyplot as plt

import joblib
import pandas as pd
import torch
import torch.nn as nn


def main():
    task_name = "load"
    if task_name == "load":
        dataset_cls = LoadDataset
        data_config = LoadDataConfigurations
        feature_config = LoadFeatureConfigurations
        criterion = nn.MSELoss()
    else:
        dataset_cls = SolarDataset
        data_config = SolarDataConfigurations
        feature_config = SolarFeatureConfigurations
        # criterion = nMAELoss(nominal=feature_config.CAPACITY)
        criterion = nn.MSELoss()

    # run_training(
    #     task_name,
    #     data_config,
    #     feature_config,
    #     criterion=criterion,
    # )
    test_dataset = dataset_cls(data_config.FILE_PATH, "", f"{task_name}_test.csv")
    X_test, y_test = test_dataset.pandas_reader_dataset(
        target=feature_config.TARGET,
        time_column="timestamp",
        timedeltas=feature_config.TIMEDELTAS,
    )
    preprocessor = joblib.load(f"artifacts/{task_name}/preprocessor.pkl")
    model = DeepResidualMLP(preprocessor.transform(X_test).shape[1])
    model.load_state_dict(torch.load(f"artifacts/{task_name}/model.pt"))
    model.eval()
    preds = predict(preprocessor, model, X_test)
    print("✅ 예측 결과:", preds[:5])

    plot_predictions(
        y_test.values.reshape(
            -1,
        ),
        preds,
        task_name=task_name,
    )


if __name__ == "__main__":

    main()
