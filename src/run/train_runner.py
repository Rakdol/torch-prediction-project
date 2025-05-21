import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.model.base_model import DeepResidualMLP
from src.dataprep.pipeline import get_input_pipeline
from src.train.trainer import train_model
from src.train.evaluator import evaluate_model
from src.train.metrics import nMAELoss
from src.train.visual import plot_loss
from src.dataprep.dataset import LoadDataset, SolarDataset, TorchDataset


def run_training(task_name, data_config, feature_config, criterion=None):
    dataset_cls = LoadDataset if task_name == "load" else SolarDataset

    train_dataset = dataset_cls(data_config.FILE_PATH, "", f"{task_name}_train.csv")
    test_dataset = dataset_cls(data_config.FILE_PATH, "", f"{task_name}_test.csv")

    X_train, y_train = train_dataset.pandas_reader_dataset(
        target=feature_config.TARGET,
        time_column="timestamp",
        timedeltas=feature_config.TIMEDELTAS,
    )
    X_test, y_test = test_dataset.pandas_reader_dataset(
        target=feature_config.TARGET,
        time_column="timestamp",
        timedeltas=feature_config.TIMEDELTAS,
    )

    # 전처리기 정의 및 학습
    preprocessor = get_input_pipeline(
        feature_config.NUM_FEATURES, feature_config.CAT_FEATURES
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_valid_scaled = preprocessor.transform(X_valid)
    X_test_scaled = preprocessor.transform(X_test)

    joblib.dump(preprocessor, f"artifacts/{task_name}/preprocessor.pkl")

    # TensorDataset 및 DataLoader 구성
    train_loader = DataLoader(TorchDataset(X_train_scaled, y_train), batch_size=256)
    val_loader = DataLoader(TorchDataset(X_valid_scaled, y_valid), batch_size=256)
    test_loader = DataLoader(TorchDataset(X_test_scaled, y_test), batch_size=256)

    model = DeepResidualMLP(X_train_scaled.shape[1])
    if criterion is not None:
        criterion = criterion
    else:
        criterion = nn.MSELoss()

    model, train_loss, val_loss = train_model(
        model,
        train_loader,
        val_loader,
        epochs=2000,
        min_delta=1e-5,
        lr=1e-2,
        patience=500,
        criterion=criterion,
        save_path=f"artifacts/{task_name}/model.pt",
    )

    plot_loss(train_loss, val_loss)
    if task_name == "solar":
        nominal = 500
    else:
        nominal = None
    eval_result = evaluate_model(model, test_loader, nominal=nominal)
    print(f"[{task_name.upper()}] Evaluation: {eval_result}")
