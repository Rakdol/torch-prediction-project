from argparse import ArgumentParser
import torch
import joblib
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from src.preprocess import get_input_pipeline
from src.model import DeepResidualMLP
from src.train import train_model
from src.evaluate import evaluate_model
from src.loader import LoadDataset, SolarDataset, TorchDataset
from src.visual import plot_train_test_target, plot_loss
from src.configurations import (
    SolarDataConfigurations,
    SolarFeatureConfigurations,
    LoadDataConfigurations,
    LoadFeatureConfigurations,
)

from pathlib import Path

PAKAGE_ROOT = Path(__name__).resolve()
print(f"Package root: {PAKAGE_ROOT}")
ARTIFACTS_ROOT = PAKAGE_ROOT / "artifacts"
print(f"Artifacts root: {ARTIFACTS_ROOT}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--task",
        default="load",
        type=str,
        choices=["load", "solar"],
        required=True,
        help="Task name (load or solar)",
    )

    args = parser.parse_args()
    task = args.task.lower()

    if task == "load":
        print("Load Data Configurations")
        print(LoadDataConfigurations.FILE_PATH)
        data_config = LoadDataConfigurations
        feature_config = LoadFeatureConfigurations
        train_datset = LoadDataset(
            upstream_directory=data_config.FILE_PATH,
            file_prefix="",
            file_name="load_train.csv",
        )
        test_dataset = LoadDataset(
            upstream_directory=data_config.FILE_PATH,
            file_prefix="",
            file_name="load_test.csv",
        )
        X_train, y_train = train_datset.pandas_reader_dataset(
            timedeltas=data_config.TIMEDELTAS, target="demand", time_column="timestamp"
        )
        X_test, y_test = test_dataset.pandas_reader_dataset(
            timedeltas=data_config.TIMEDELTAS, target="demand", time_column="timestamp"
        )

        print(f"Shape of X_train {X_train.shape} and y_train {y_train.shape}")
        print(f"Shape of X_test {X_test.shape} and y_test {y_test.shape}")
        plot_train_test_target(X_train, y_train, X_test, y_test, task)
        print("CUDA 사용 가능 여부:", torch.cuda.is_available())
        print("현재 사용 중인 디바이스:", torch.cuda.current_device())
        print("디바이스 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))

        numerical_features = feature_config.NUM_FEATURES
        categorical_features = feature_config.CAT_FEATURES
        print(f"Numerical features: {numerical_features}")
        print(f"Categorical features: {categorical_features}")
        print(f"Target: {feature_config.TARGET}")

        preprocessor = get_input_pipeline(numerical_features, categorical_features)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_valid_scaled = preprocessor.transform(X_valid)
        X_test_scaled = preprocessor.transform(X_test)

        print(
            f"Shape of X_train_scaled {X_train_scaled.shape} and y_train {y_train.shape}"
        )
        print(
            f"Shape of X_valid_scaled {X_valid_scaled.shape} and y_valid {y_valid.shape}"
        )
        print(f"Shape of X_test_scaled {X_test_scaled.shape} and y_test {y_test.shape}")
        # Save the preprocessor
        joblib.dump(preprocessor, ARTIFACTS_ROOT / task / "preprocessor.pkl")
        print(f"Preprocessor saved to {ARTIFACTS_ROOT / task / 'preprocessor.pkl'}")

        # Create datasets
        train_dataset = TorchDataset(X_train_scaled, y_train)
        valid_dataset = TorchDataset(X_valid_scaled, y_valid)
        test_dataset = TorchDataset(X_test_scaled, y_test)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=256)
        val_loader = DataLoader(valid_dataset, batch_size=256)
        test_loader = DataLoader(test_dataset, batch_size=256)

        model = DeepResidualMLP(input_dim=X_train_scaled.shape[1])

        save_path = ARTIFACTS_ROOT / task / "best_model.pt"
        model, train_loss_history, val_loss_history = train_model(
            model,
            train_loader,
            val_loader,
            epochs=2000,
            min_delta=1e-5,
            lr=1e-2,
            patience=500,
            save_path=save_path,
        )
        plot_loss(train_loss_history, val_loss_history)
        print(f"Model saved to {save_path}")

        evaluations = evaluate_model(
            model,
            test_loader,
        )

        print(f"Evaluation metrics: {evaluations}")

        print("Training completed successfully.")
    elif task == "solar":
        
