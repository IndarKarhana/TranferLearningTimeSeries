import pytest
from darts.metrics import smape
from darts import TimeSeries
import numpy as np
import warnings
from unittest.mock import MagicMock, patch
from src.model import train_model  

@patch('darts.models.NBEATSModel')
def test_train_model(MockNBEATSModel):
    mock_model = MockNBEATSModel()
    
    train_series = [MagicMock(spec=TimeSeries)]
    val_series = [MagicMock(spec=TimeSeries)]

    mock_model.fit.return_value = None

    trained_model = train_model(mock_model, train_series, val_series, epochs=5, verbose=False)

    mock_model.fit.assert_called_with(train_series, epochs=5, val_series=val_series, verbose=False)
    assert trained_model == mock_model