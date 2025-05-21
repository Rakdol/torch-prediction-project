import argparse
from src.config.task_configs import (
    LoadDataConfigurations,
    LoadFeatureConfigurations,
    SolarDataConfigurations,
    SolarFeatureConfigurations,
)
from src.run.train_runner import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["load", "solar"])
    args = parser.parse_args()

    if args.task == "load":
        run_training("load", LoadDataConfigurations, LoadFeatureConfigurations)
    else:
        run_training("solar", SolarDataConfigurations, SolarFeatureConfigurations)
