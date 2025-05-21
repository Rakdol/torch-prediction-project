from pathlib import Path


import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


from src.features import weather_features as wtr
from src.features import time_features as tr


class LoadDataset(object):
    def __init__(
        self,
        upstream_directory: str,
        file_prefix: str,
        file_name: str,
    ):

        self.upstream_directory = upstream_directory
        self.file_prefix = file_prefix
        self.file_name = file_name

    def pandas_reader_dataset(
        self, target: str, time_column: str | None, timedeltas: list[str] | None
    ) -> tuple[pd.DataFrame, pd.Series]:
        file_paths = str(
            Path() / self.upstream_directory / self.file_prefix / self.file_name
        )
        df = pd.read_csv(file_paths)
        if time_column is not None:
            df_ = self.transform_process(df, timedeltas, time_column, target)
            X = df_.drop(labels=[target], axis=1)
            y = df_[target]
            return X, y

        X = df.drop(labels=[target], axis=1)
        y = df[target]

        return X, y

    def transform_process(
        self,
        df: pd.DataFrame,
        timedeltas: list[str] | None,
        time_column: str = "timestamp",
        target: str = "demand",
    ) -> pd.DataFrame:
        tr.set_time_index(df, time_column)
        df = tr.create_time_features(df)

        df = tr.create_time_lag_features(df, target=target, timedeltas=timedeltas)
        hour_group_energy = tr.grouped_frame(
            df=df, group_col_list=["hour"], target_col_list=[target], method="mean"
        )
        hour_group_energy_std = tr.grouped_frame(
            df=df, group_col_list=["hour"], target_col_list=[target], method="std"
        )
        dayweek_hour_gruop_energy = tr.grouped_frame(
            df=df,
            group_col_list=["dayofweek", "hour"],
            target_col_list=[target],
            method="mean",
        )
        df["hour_mean"] = df.apply(
            lambda x: hour_group_energy.loc[
                (hour_group_energy.hour == x["hour"]), f"{target}_mean"
            ].values[0],
            axis=1,
        )
        df["hour_std"] = df.apply(
            lambda x: hour_group_energy_std.loc[
                (hour_group_energy_std.hour == x["hour"]), f"{target}_std"
            ].values[0],
            axis=1,
        )
        df["dayweek_hour_mean"] = df.apply(
            lambda x: dayweek_hour_gruop_energy.loc[
                (dayweek_hour_gruop_energy.hour == x["hour"])
                & (dayweek_hour_gruop_energy.dayofweek == x["dayofweek"]),
                f"{target}_mean",
            ].values[0],
            axis=1,
        )

        return df


class SolarDataset(object):
    def __init__(
        self,
        upstream_directory: str,
        file_prefix: str,
        file_name: str,
    ):

        self.upstream_directory = upstream_directory
        self.file_prefix = file_prefix
        self.file_name = file_name

    def pandas_reader_dataset(
        self, target: str, time_column: str | None, timedeltas: list[str] | None
    ) -> tuple[pd.DataFrame, pd.Series]:
        file_paths = str(
            Path() / self.upstream_directory / self.file_prefix / self.file_name
        )
        df = pd.read_csv(file_paths)
        if time_column is not None:
            df_ = self.transform_process(df, timedeltas, time_column, target)
            X = df_.drop(labels=[target], axis=1)
            y = df_[target]
            return X, y

        X = df.drop(labels=[target], axis=1)
        y = df[target]

        return X, y

    def transform_process(
        self,
        df: pd.DataFrame,
        timedeltas: list[str] | None,
        time_column: str = "timestamp",
        target: str = "generation",
    ) -> pd.DataFrame:

        tr.set_time_index(df, time_column)
        df = tr.create_time_features(df)
        df = tr.create_time_lag_features(df, target=target, timedeltas=timedeltas)

        hour_group_energy = tr.grouped_frame(
            df=df, group_col_list=["hour"], target_col_list=[target], method="mean"
        )
        hour_group_energy_std = tr.grouped_frame(
            df=df, group_col_list=["hour"], target_col_list=[target], method="std"
        )
        cloud_hour_gruop_energy = tr.grouped_frame(
            df=df,
            group_col_list=["cloud_cover", "hour"],
            target_col_list=[target],
            method="mean",
        )
        df["hour_mean"] = df.apply(
            lambda x: hour_group_energy.loc[
                (hour_group_energy.hour == x["hour"]), f"{target}_mean"
            ].values[0],
            axis=1,
        )
        df["hour_std"] = df.apply(
            lambda x: hour_group_energy_std.loc[
                (hour_group_energy_std.hour == x["hour"]), f"{target}_std"
            ].values[0],
            axis=1,
        )

        df["cloud_hour_mean"] = df.apply(
            lambda x: cloud_hour_gruop_energy.loc[
                (cloud_hour_gruop_energy.hour == x["hour"])
                & (cloud_hour_gruop_energy["cloud_cover"] == x["cloud_cover"]),
                f"{target}_mean",
            ].values[0],
            axis=1,
        )

        df = tr.transform_cyclic(df, col="hour", max_val=23)
        df = tr.transform_cyclic(df, col="month", max_val=12)
        df = tr.transform_cyclic(df, col="dayofweek", max_val=6)
        df = tr.transform_cyclic(df, col="quarter", max_val=4)
        df = tr.transform_cyclic(df, col="dayofyear", max_val=365)
        df = tr.transform_cyclic(df, col="dayofmonth", max_val=31)
        df = wtr.convert_wind(df=df, speed="wind_speed", direction="wind_direction")
        # df = wtr.convert_cloudy(df=df, column="cloud_cover", Forecast=False)

        return df


class TorchDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
