import os
from pathlib import Path
from logging import getLogger

PAKAGE_ROOT = Path(__name__).resolve().parents[0]
print(f"Package root: {PAKAGE_ROOT}")


class SolarDataConfigurations:
    FILE_PATH = os.path.join(PAKAGE_ROOT, "data/preprocess/solar")
    TRAIN_FILE = os.path.join(FILE_PATH, "solar_train.csv")
    TEST_FILE = os.path.join(FILE_PATH, "solar_test.csv")


class LoadDataConfigurations:

    FILE_PATH = os.path.join(PAKAGE_ROOT, "data/preprocess/load")
    TRAIN_FILE = os.path.join(FILE_PATH, "load_train.csv")
    TEST_FILE = os.path.join(FILE_PATH, "load_test.csv")


class SolarFeatureConfigurations:
    TARGET = "generation"
    ID_FEATURES = ["solar_id"]

    NUM_FEATURES = [
        "temperature",
        "humidity",
        "wind_speed",
        "wind_direction",
    ]

    CAT_FEATURES = ["cloud_cover"]
    DATE_FEATURES = ["timestamp"]


class LoadFeatureConfigurations:
    TARGET = "demand"

    ID_FEATURES = ["load_id"]

    NUM_FEATURES = [
        "temperature",
        "humidity",
        "hour_mean",
        "hour_std",
        "dayweek_hour_mean",
    ]

    TIMEDELTAS = ["1 days", "2 days", "3 days", "7 days"]
    CAT_FEATURES = []

    DATE_FEATURES = ["timestamp"]
