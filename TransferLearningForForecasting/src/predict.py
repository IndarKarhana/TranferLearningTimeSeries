# src/predict.py

import pandas as pd
import joblib
import yaml
import numpy as np
from darts import TimeSeries
from src.data_processing import replace_outliers, apply_smoothing, create_time_series, scale_series
from src.model import load_model

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config("TransferLearningForForecasting/configs/config.yaml")


def get_forecasts(model, series_list: list, series_names: list, horizon: int) -> list:
    """
    Generate forecasts using the trained model.

    Args:
        model: The trained Darts model.
        series_list (list): List of TimeSeries objects for prediction.
        series_names (list): List of series names corresponding to the TimeSeries objects.
        horizon (int): Forecasting horizon.

    Returns:
        list: List of tuples where each tuple contains the predicted TimeSeries and its name.
    """
    forecasts = model.predict(series=series_list, n=horizon)
    return list(zip(forecasts, series_names))

def inverse_scale_forecasts(forecasts: list, scaler) -> list:
    """
    Apply inverse scaling to the forecasts.

    Args:
        forecasts (list): List of tuples where each tuple contains a forecasted TimeSeries and its name.
        scaler: Scaler object used for scaling the data.

    Returns:
        list: List of tuples where each tuple contains the unscaled TimeSeries and its name.
    """
    return [(scaler.inverse_transform(forecast), series_name) for forecast, series_name in forecasts]

def save_forecasts(forecasts: list, config: dict):
    """
    Save the forecasts to a CSV file.

    Args:
        forecasts (list): List of tuples where each tuple contains the forecasted TimeSeries and its name.
        config (dict): Configuration dictionary.
    """
    save_path = config["output"]["prediction_save_path"]
    
    # Extract forecasts and series names from the tuples
    df_list = []
    for forecast, series_name in forecasts:
        df = forecast.pd_dataframe().reset_index()
        df["Series_Name"] = series_name  # Add the series name column
        df_list.append(df)
    
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_csv(save_path, index=False)
    print(f"Forecasts saved to {save_path}")

def run_prediction(pred_series,series_name,config: dict):
    """
    Main function to run the prediction workflow.

    Args:
        config (dict): Configuration dictionary.
    """

    # Load the model
    model_path = config["output"]["fine_tune_model_save_path"]
    model = load_model(model_path, config["model"])

    # Generate forecasts
    horizon = config["model"]["horizon"]
    forecasts = get_forecasts(model, pred_series, series_name, horizon)

    # Inverse scale the forecasts if scaling was applied
    if config["model"]["scale"]:
        # Load the saved scaler (assumed to be saved during training)
        scaler = joblib.load(config["output"]["scaler_save_path"])
        forecasts = inverse_scale_forecasts(forecasts, scaler)

    # Save the forecasts
    save_forecasts(forecasts, config)
