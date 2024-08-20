import pandas as pd

def ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Ensures that the specified column is in datetime format.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        date_col (str): The column name that should be in datetime format.
    
    Returns:
        pd.DataFrame: Dataframe with the column converted to datetime format.
    
    Raises:
        ValueError: If the column cannot be converted to datetime.
    """
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='raise')
    except Exception as e:
        raise ValueError(f"Error converting column '{date_col}' to datetime format: {e}")
    
    return df

def validate_config(df: pd.DataFrame, config: dict):
    """
    Validates the configuration by checking if the required columns exist in the dataframe.
    
    Args:
        df (pd.DataFrame): The dataframe to validate.
        config (dict): The configuration dictionary.
    
    Raises:
        ValueError: If any required column is missing from the dataframe.
    """
    required_columns = [config["data"]["group_col"], config["data"]["target_col"], config["data"]["date_col"]]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the DataFrame.")