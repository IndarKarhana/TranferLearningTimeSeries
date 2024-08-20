import time
import numpy as np
import matplotlib.pyplot as plt
from darts.metrics import smape
from darts.models import NBEATSModel
from darts import TimeSeries
import tqdm as tq
from typing import List, Tuple

def train_model(model, train_series, val_series, epochs, verbose=True) -> NBEATSModel:
    """
    Train the NBEATS model on training data.

    Args:
        model (NBEATSModel): The NBEATS model to be trained.
        train_series (List[TimeSeries]): List of training TimeSeries.
        val_series (List[TimeSeries]): List of validation TimeSeries.
        epochs (int): Number of epochs to train.
        verbose (bool): Whether to print training progress.

    Returns:
        NBEATSModel: The trained model.
    """
    model.fit(train_series, epochs=epochs, val_series=val_series, verbose=verbose)
    return model

def load_model(model_path: str,cfg) -> NBEATSModel: #removed cfg
    """
    Load a pre-trained NBEATS model and apply device configurations.

    Args:
        model_path (str): Path to the pre-trained model.
        cfg (dict): Configuration dictionary for initializing the model.

    Returns:
        NBEATSModel: The loaded NBEATS model.
    """
    # model = initialize_model(cfg)
    model = NBEATSModel.load(model_path)
    print("Model loaded.")
    return model

