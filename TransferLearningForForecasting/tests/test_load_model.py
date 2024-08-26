import pytest
from darts.metrics import smape
from darts import TimeSeries
import numpy as np
import warnings
import yaml
from darts.models import NBEATSModel
from unittest.mock import patch, MagicMock
from src.model import load_model  

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config("TransferLearningForForecasting/configs/config.yaml")

@patch('darts.models.NBEATSModel.load')
def test_load_model(mock_load):
    mock_model = MagicMock(spec=NBEATSModel)
    mock_load.return_value = mock_model
    
    model_path = config["model"]["path"]
    cfg = {}  # Assuming cfg is not used anymore

    loaded_model = load_model(model_path, cfg)

    mock_load.assert_called_with(model_path)
    assert loaded_model == mock_model