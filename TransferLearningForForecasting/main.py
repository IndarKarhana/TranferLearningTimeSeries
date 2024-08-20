import pandas as pd
import yaml
import joblib
from src.data_processing import replace_outliers, apply_smoothing, create_time_series, train_test_split, scale_series
from src.model import load_model, train_model, eval_local_model
from src.predict import run_prediction

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration file
    config = load_config("configs/config.yaml")

    # Load the dataframe
    data_path = config["data"]["path"]
    workable_df = pd.read_csv(data_path)
    
    # Extract configuration values
    group_col = config["data"]["group_col"]
    target_col = config["data"]["target_col"]
    date_col = config["data"]["date_col"]
    smoothed_col = config["data"].get("smoothed_col", "SmoothedTotalQuantity")
    TimeSeriesName_col = config["data"].get("TimeSeriesName_col", group_col)
    min_size = config["model"]["min_size"]
    horizon = config["model"]["horizon"]

    # Preprocessing steps based on config
    if config["preprocessing"]["replace_outliers"]:
        workable_df = replace_outliers(workable_df, group_col=group_col, target_col=target_col)
    
    if config["preprocessing"]["smoothing"]["apply"]:
        window_size = config["preprocessing"]["smoothing"].get("window_size", 3)
        workable_df = apply_smoothing(workable_df, group_col=group_col, target_col=target_col, smoothed_col=smoothed_col, window_size=window_size)
    
    # Create time series
    all_series_ret = create_time_series(
        workable_df, date_col=date_col, value_col=smoothed_col, TimeSeriesName_col=TimeSeriesName_col, group_col=group_col, min_size=min_size
    )
    # Perform train/test split
    ret_train, ret_test = train_test_split(all_series_ret, horizon=horizon)

    # Scale the data if specified
    if config["model"]["scale"]:
        ret_train_scaled, ret_test_scaled, scaler_ret = scale_series(ret_train, ret_test)
        # Save the scaler for later inverse scaling during prediction
        joblib.dump(scaler_ret, config["output"]["scaler_save_path"])
        print("Scaling applied and scaler saved.")
    else:
        ret_train_scaled, ret_test_scaled = ret_train, ret_test
        print("Skipping scaling.")

    print("Processing completed successfully.")

    # Conditional retraining based on config
    if config.get("model").get("retrain", False):
        # Load the model
        model_path = config["model"]["path"]
        nbeats_model = load_model(model_path)
        # nbeats_model = load_model(model_path, config["model"])

        # Train the model
        epochs = config["model"]["epochs"]
        nbeats_model = train_model(nbeats_model, ret_train_scaled, ret_test_scaled, epochs, config["model"]["verbose"])

        # Evaluate the model
        smapes, elapsed_time = eval_local_model(ret_train_scaled, ret_test_scaled, horizon, nbeats_model, cfg=config)
        print(f"Model evaluation completed in {elapsed_time:.2f} seconds.")
        print(f'smapes:', smapes)

    # Conditional prediction based on config
    if config.get("data").get("prediction_path"):
        run_prediction(config)

if __name__ == "__main__":
    main()