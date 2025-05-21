import pandas as pd
import numpy as np


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
