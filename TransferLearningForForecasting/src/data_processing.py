# src/data_processing.py

import pandas as pd
from typing import List, Tuple
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
from src.utils import ensure_datetime

def replace_outliers(df: pd.DataFrame, group_col: str, target_col: str, threshold: float = 3) -> pd.DataFrame:
    """
    Detects and replaces outliers in a grouped time series using the mean and standard deviation.
    
    Args:
        df (pd.DataFrame): Input dataframe containing time series data.
        group_col (str): Column name to group by (e.g., 'StockCode').
        target_col (str): Column name where outliers are detected (e.g., 'TotalQuantity').
        threshold (float): The threshold multiplier for outlier detection (default is 3).
    
    Returns:
        pd.DataFrame: Dataframe with outliers replaced by the mean value.
    """
    def _replace_outliers(series: pd.Series) -> pd.Series:
        mean_val = series.mean()
        std_dev = series.std()
        outliers = abs(series - mean_val) > (threshold * std_dev)
        series[outliers] = mean_val
        return series

    df[target_col] = df.groupby(group_col)[target_col].transform(_replace_outliers)
    return df

def apply_smoothing(df: pd.DataFrame, group_col: str, target_col: str, smoothed_col: str, window_size: int = 3) -> pd.DataFrame:
    """
    Applies a rolling average to smooth a time series.
    
    Args:
        df (pd.DataFrame): Input dataframe containing time series data.
        group_col (str): Column name to group by (e.g., 'StockCode').
        target_col (str): Column name to apply smoothing on (e.g., 'TotalQuantity').
        smoothed_col (str): The name of the new column to store the smoothed values.
        window_size (int): The window size for the rolling average (default is 3).
    
    Returns:
        pd.DataFrame: Dataframe with smoothed values in the specified column.
    """
    df[smoothed_col] = df.groupby(group_col)[target_col].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    return df

def train_test_split(series_list: List[TimeSeries], horizon: int) -> Tuple[List[TimeSeries], List[TimeSeries]]:
    """
    Splits the time series data into training and test sets.
    
    Args:
        series_list (List[TimeSeries]): List of time series to split.
        horizon (int): Number of points to use for the test set.
    
    Returns:
        Tuple: Two lists, one for training and one for testing.
    """
    ret_train = [s[:-horizon] for s in series_list]
    ret_test = [s[-horizon:] for s in series_list]
    return ret_train, ret_test

# Convert TimeSeries data to float32
def convert_timeseries_to_float32(ts_list):
    return [ts.astype(np.float32) for ts in ts_list]

def scale_series(train_series: List[TimeSeries], test_series: List[TimeSeries]) -> Tuple[List[TimeSeries], List[TimeSeries], Scaler]:
    """
    Scales the time series data using MaxAbsScaler.
    
    Args:
        train_series (List[TimeSeries]): Training time series.
        test_series (List[TimeSeries]): Testing time series.
    
    Returns:
        Tuple: Scaled training and testing series along with the scaler object.
    """
    scaler = Scaler(scaler=MaxAbsScaler())
    train_scaled = scaler.fit_transform(train_series)
    test_scaled = scaler.transform(test_series)
    train_scaled = convert_timeseries_to_float32(train_scaled)
    test_scaled = convert_timeseries_to_float32(test_scaled)

    return train_scaled, test_scaled, scaler

def create_time_series(df: pd.DataFrame, date_col: str, value_col: str, TimeSeriesName_col: str, group_col: str, min_size: int) -> List[TimeSeries]:
    # Ensure the date column is in datetime format
    df = ensure_datetime(df, date_col)

    all_series = []
    unique_time_series = df[TimeSeriesName_col].unique()
    
    for name in unique_time_series:
        series_df = df.loc[df[group_col] == name]
        time_series = TimeSeries.from_dataframe(
            series_df[[date_col, value_col]].set_index(date_col).sort_index()
        )
        if len(time_series) > min_size:
            all_series.append(time_series)
    return all_series


def scale_series_train(train_series: List[TimeSeries]) -> Tuple[List[TimeSeries], Scaler]:
    """
    Scales the time series data using MaxAbsScaler.
    
    Args:
        train_series (List[TimeSeries]): Training time series.
    
    Returns:
        Tuple: Scaled training  along with the scaler object.
    """
    scaler = Scaler(scaler=MaxAbsScaler())
    train_scaled = scaler.fit_transform(train_series)
    train_scaled = convert_timeseries_to_float32(train_scaled)

    return train_scaled, scaler


