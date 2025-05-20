from pathlib import Path

PAKAGE_ROOT = Path(__file__).resolve().parents[0]
FIGURE_ROOT = PAKAGE_ROOT / "artifacts" / "figures"

import numpy as np
import matplotlib.pyplot as plt


def plot_train_test_target(y_train: np.ndarray, y_test: np.ndarray, task: str):
    """
    Plot the training and test data.

    Parameters
    ----------
    y_train : np.ndarray
        The training data.
    y_test : np.ndarray
        The test data.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14

    fig, axes = plt.subplots(1, 1, figsize=(15, 4))

    train_load = y_train
    test_load = y_test

    indices = list(range(train_load.shape[0] + test_load.shape[0]))

    axes.plot(indices[: train_load.shape[0]], train_load, label="train set")
    axes.plot(indices[train_load.shape[0] :], test_load, label="test set")
    axes.legend()
    axes.set_xlabel("Time")
    axes.set_ylabel(task)
    axes.set_title(f"{task} Train and Test Data")
    plt.savefig(FIGURE_ROOT / f"{task}_train_test.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_loss(train_loss_history: list[float], val_loss_history: list[float]):
    """
    Plot the training and validation loss.

    Args:
        train_loss_history (list): training loss history
        val_loss_history (list): validation loss history
    """

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(FIGURE_ROOT / "train_val_loss.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
