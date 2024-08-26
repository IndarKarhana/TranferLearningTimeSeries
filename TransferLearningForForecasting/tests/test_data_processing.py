import pytest
import pandas as pd
import numpy as np
from darts import TimeSeries
from src.data_processing import (
    replace_outliers, apply_smoothing, train_test_split, scale_series,
    create_time_series, scale_series_train
)
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MaxAbsScaler
from datetime import datetime, timedelta

@pytest.fixture
def sample_df():
    """Fixture for generating sample DataFrame for testing."""
    data = {
        "StockCode": ["A"] * 20 + ["B"] * 20,
        "InvoiceDate": [
            f"2023-01-{i+1:02d}" for i in range(10)] * 2  # Dates repeated for two groups
            + [f"2023-02-{i+1:02d}" for i in range(10)] * 2,
        "TotalQuantity": [
            10, 12, 14, 13, 9, 11, 500, 14, 13, 12,  # Group A, normal values with one outlier
            15, 17, 20, 13, 14, 16, 50, 12, 14, 11,  # Group A continuation with another outlier
            5, 7, 6, 8, 9, 7, 6, 1000, 7, 9,  # Group B, normal values with an outlier
            8, 7, 9, 6, 8, 7, 6, 8, 7, 6,  # Group B continuation
        ]
    }
    df = pd.DataFrame(data)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def test_replace_outliers(sample_df):
    """Test the replace_outliers function."""
    processed_df = replace_outliers(sample_df, group_col="StockCode", target_col="TotalQuantity", threshold=2)
    
    # Check if outliers have been replaced
    assert processed_df.loc[processed_df["StockCode"] == "A", "TotalQuantity"].iloc[6] != 500, "Outlier not replaced correctly."
    assert processed_df.loc[processed_df["StockCode"] == "B", "TotalQuantity"].iloc[7] != 1000, "Outlier not replaced correctly."

def test_apply_smoothing(sample_df):
    """Test the apply_smoothing function."""
    smoothed_df = apply_smoothing(sample_df, group_col="StockCode", target_col="TotalQuantity", smoothed_col="SmoothedQuantity", window_size=2)
    
    # Check if the smoothing is applied correctly
    assert smoothed_df["SmoothedQuantity"].iloc[1] == 11, "Smoothing not applied correctly for StockCode A."
    assert smoothed_df["SmoothedQuantity"].iloc[5] == 10, "Smoothing not applied correctly for StockCode B."

def test_train_test_split():
    """Test the train_test_split function."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    values = np.arange(10)
    ts = TimeSeries.from_times_and_values(dates, values)
    
    train, test = train_test_split([ts], horizon=3)
    
    assert len(train[0]) == 7, "Train split is not of expected length."
    assert len(test[0]) == 3, "Test split is not of expected length."

def test_scale_series():
    """Test the scaling function."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    values = np.arange(10).reshape(-1, 1)
    ts = TimeSeries.from_times_and_values(dates, values)

    train_scaled, test_scaled, scaler = scale_series([ts], [ts[:-3]])

    assert isinstance(scaler, Scaler), "Scaler object not returned correctly."
    assert all([s.values().min() >= -1 and s.values().max() <= 1 for s in train_scaled]), "Training series not scaled correctly."
    assert all([s.values().min() >= -1 and s.values().max() <= 1 for s in test_scaled]), "Testing series not scaled correctly."

def test_scale_series_train():
    """Test the scaling function for training data only."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    values = np.arange(10).reshape(-1, 1)
    ts = TimeSeries.from_times_and_values(dates, values)

    train_scaled, scaler = scale_series_train([ts])

    assert isinstance(scaler, Scaler), "Scaler object not returned correctly."
    assert all([s.values().min() >= -1 and s.values().max() <= 1 for s in train_scaled]), "Training series not scaled correctly."
