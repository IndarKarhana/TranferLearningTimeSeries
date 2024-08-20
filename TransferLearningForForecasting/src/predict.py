# src/predict.py

import pandas as pd
import joblib
from darts import TimeSeries
from src.data_processing import replace_outliers, apply_smoothing, create_time_series, scale_series
from src.model import load_model

def load_prediction_data(file_path: str, config: dict) -> pd.DataFrame:
    """
    Load and preprocess prediction data.

    Args:
        file_path (str): Path to the new dataset for prediction.
        config (dict): Configuration dictionary.

    Returns:
        pd.DataFrame: Preprocessed prediction dataframe.
    """
    df = pd.read_csv(file_path)

    # Extract configuration values
    group_col = config["data"]["group_col"]
    target_col = config["data"]["target_col"]
    date_col = config["data"]["date_col"]
    smoothed_col = config["data"].get("smoothed_col", "SmoothedTotalQuantity")

    # Preprocessing steps based on config
    if config["preprocessing"]["replace_outliers"]:
        df = replace_outliers(df, group_col=group_col, target_col=target_col)

    if config["preprocessing"]["smoothing"]["apply"]:
        window_size = config["preprocessing"]["smoothing"].get("window_size", 3)
        df = apply_smoothing(df, group_col=group_col, target_col=target_col, smoothed_col=smoothed_col, window_size=window_size)

    return df

def get_forecasts(model, series_list: list, horizon: int) -> list:
    """
    Generate forecasts using the trained model.

    Args:
        model: The trained Darts model.
        series_list (list): List of TimeSeries objects for prediction.
        horizon (int): Forecasting horizon.

    Returns:
        list: List of predicted TimeSeries.
    """
    forecasts = [model.predict(n=horizon, series=s) for s in series_list]
    return forecasts

def inverse_scale_forecasts(forecasts: list, scaler) -> list:
    """
    Apply inverse scaling to the forecasts.

    Args:
        forecasts (list): List of forecasted TimeSeries.
        scaler: Scaler object used for scaling the data.

    Returns:
        list: List of unscaled TimeSeries.
    """
    return [scaler.inverse_transform(forecast) for forecast in forecasts]

def save_forecasts(forecasts: list, config: dict):
    """
    Save the forecasts to a CSV file.

    Args:
        forecasts (list): List of forecasted TimeSeries.
        config (dict): Configuration dictionary.
    """
    save_path = config["output"]["prediction_save_path"]
    df_list = [forecast.pd_dataframe().reset_index() for forecast in forecasts]
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_csv(save_path, index=False)
    print(f"Forecasts saved to {save_path}")

def run_prediction(config: dict):
    """
    Main function to run the prediction workflow.

    Args:
        config (dict): Configuration dictionary.
    """
    # Load and preprocess prediction data
    pred_data_path = config["data"]["prediction_path"]
    pred_df = load_prediction_data(pred_data_path, config)

    # Create time series
    group_col = config["data"]["group_col"]
    smoothed_col = config["data"]["smoothed_col"]
    date_col = config["data"]["date_col"]
    TimeSeriesName_col = config["data"]["TimeSeriesName_col"]
    min_size = config["model"]["min_size"]

    pred_series = create_time_series(
        pred_df, date_col=date_col, value_col=smoothed_col, TimeSeriesName_col=TimeSeriesName_col, group_col=group_col, min_size=min_size
    )

    # Load the model
    model_path = config["model"]["path"]
    model = load_model(model_path, config["model"])

    # Generate forecasts
    horizon = config["model"]["horizon"]
    forecasts = get_forecasts(model, pred_series, horizon)

    # Inverse scale the forecasts if scaling was applied
    if config["model"]["scale"]:
        # Load the saved scaler (assumed to be saved during training)
        scaler = joblib.load(config["output"]["scaler_save_path"])
        forecasts = inverse_scale_forecasts(forecasts, scaler)

    # Save the forecasts
    save_forecasts(forecasts, config)