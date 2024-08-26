import pytest
import pandas as pd
import numpy as np
from darts import TimeSeries
from src.utils import ensure_datetime, validate_config, eval_forecasts

@pytest.fixture
def sample_df():
    """Fixture for generating a sample DataFrame for testing."""
    data = {
        "StockCode": ["A", "A", "A", "B", "B", "B"],
        "InvoiceDate": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-01", "2023-01-02", "2023-01-03"],
        "TotalQuantity": [10, 12, 15, 20, 25, 30]
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def sample_config():
    """Fixture for generating a sample config dictionary for testing."""
    return {
        "data": {
            "group_col": "StockCode",
            "target_col": "TotalQuantity",
            "date_col": "InvoiceDate"
        }
    }

def test_ensure_datetime(sample_df):
    """Test the ensure_datetime function."""
    df = ensure_datetime(sample_df, "InvoiceDate")
    
    # Check if the column is converted to datetime
    assert pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]), "InvoiceDate column is not in datetime format."
    
    # Test with invalid date format
    sample_df["InvalidDate"] = ["invalid_date"] * len(sample_df)
    with pytest.raises(ValueError, match="Error converting column 'InvalidDate' to datetime format"):
        ensure_datetime(sample_df, "InvalidDate")

def test_validate_config(sample_df, sample_config):
    """Test the validate_config function."""
    # Check if the function passes when all required columns are present
    try:
        validate_config(sample_df, sample_config)
    except ValueError:
        pytest.fail("validate_config raised ValueError unexpectedly.")
    
    # Test missing column
    sample_config["data"]["group_col"] = "MissingColumn"
    with pytest.raises(ValueError, match="Required column 'MissingColumn' is missing from the DataFrame."):
        validate_config(sample_df, sample_config)
