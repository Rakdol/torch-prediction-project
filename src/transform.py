import numpy as np
import pandas as pd


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["dayofyear"] = df.index.dayofyear
    df["dayofmonth"] = df.index.day

    return df


def create_time_lag_features(
    df: pd.DataFrame,
    target: str = "Load",
    timedeltas: list[str] = ["1 days", "2 days", "3 days", "7 days"],
) -> pd.DataFrame:
    target_map = df[target].to_dict()

    for tdelta in timedeltas:
        df["lag_" + tdelta] = (df.index - pd.Timedelta(tdelta)).map(target_map)

    return df


def transform_cyclic(df: pd.DataFrame, col: str, max_val: int) -> pd.DataFrame:
    """
    Add Cyclic featture to the dataframe
    """
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    df.pop(col)

    return df


def set_time_index(df: pd.DataFrame, time_column: str = "Forecast_time") -> None:
    df.set_index(pd.to_datetime(df[time_column]), inplace=True)
    df.drop(labels=[time_column], axis=1, inplace=True)


def grouped_frame(
    df: pd.DataFrame, group_col_list: list, target_col_list: list, method="mean"
):

    if method == "mean":
        mean_list = []
        for target in target_col_list:
            mean_list.append(target + "_mean")
        mean = df.groupby(group_col_list)[target_col_list].mean().reset_index()
        mean.columns = group_col_list + mean_list
        return mean

    elif method == "std":
        std_list = []
        for target in target_col_list:
            std_list.append(target + "_std")
        std = df.groupby(group_col_list)[target_col_list].std().reset_index()
        std.columns = group_col_list + std_list
        return std


def convert_wind(df: pd.DataFrame, speed: str, direction: str):

    df = df.copy()
    wv = df.pop(speed)
    wd_rad = df.pop(direction) * np.pi / 180

    # Calculate the wind x and y components.
    df["wind_x"] = wv * np.cos(wd_rad)
    df["wind_y"] = wv * np.sin(wd_rad)

    return df


def convert_cloudy(df: pd.DataFrame, column: str, Forecast: bool = False):
    df = df.copy()
    cloudy = df[column].copy()

    if not Forecast:
        for i in range(len(cloudy)):
            if cloudy.iloc[i] <= 5:
                cloudy.iloc[i] = "Clear"
            elif cloudy.iloc[i] <= 8:
                cloudy.iloc[i] = "Cloudy"
            else:
                cloudy.iloc[i] = "Mostly"
    else:
        for i in range(len(cloudy)):
            if cloudy.iloc[i] <= 2:
                cloudy.iloc[i] = "Clear"
            elif cloudy.iloc[i] <= 3:
                cloudy.iloc[i] = "Cloudy"
            else:
                cloudy.iloc[i] = "Mostly"
    df[column] = cloudy

    return df
