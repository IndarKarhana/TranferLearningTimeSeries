import time
import numpy as np
import matplotlib.pyplot as plt
from darts.metrics import smape
from darts.models import NBEATSModel
from darts import TimeSeries
import tqdm as tq
from typing import List, Tuple

# def eval_forecasts(pred_series: List[TimeSeries], test_series: List[TimeSeries], cfg) -> List[float]:
#     """
#     Evaluate forecasts using sMAPE metric and plot histogram of sMAPEs.

#     Args:
#         pred_series (List[TimeSeries]): List of predicted TimeSeries.
#         test_series (List[TimeSeries]): List of actual TimeSeries.
#         cfg (dict): Configuration dictionary.

#     Returns:
#         List[float]: List of sMAPE values for each series.
#     """
#     print('Computing sMAPEs...')
#     smapes = smape(test_series, pred_series)
#     mean, std = np.round(np.mean(smapes), 4), np.round(np.std(smapes), 4)
#     print(f'Avg sMAPE: {mean:.3f} ± {std:.3f}')
    
#     plt.figure(figsize=(cfg["img_dim1"], cfg["img_dim2"]), dpi=144)
#     plt.hist(smapes, bins=50)
#     plt.ylabel('Count')
#     plt.xlabel('sMAPE')
#     plt.show()
#     plt.close()
#     return smapes

# def eval_local_model(train_series: List[TimeSeries], test_series: List[TimeSeries], horizon: int, model_cls, **kwargs) -> Tuple[List[float], float]:
#     """
#     Evaluate a model on multiple time series using a specified forecasting horizon.

#     Args:
#         train_series (List[TimeSeries]): List of training TimeSeries.
#         test_series (List[TimeSeries]): List of test TimeSeries.
#         horizon (int): Forecasting horizon.
#         model_cls: The class of the model to be evaluated.
#         **kwargs: Additional configuration options.

#     Returns:
#         Tuple[List[float], float]: List of sMAPEs and elapsed time.
#     """
#     preds = []
#     start_time = time.time()
#     for series in tq.tqdm(train_series):
#         model = model_cls #(**kwargs)
#         model.fit(series)
#         pred = model.predict(n=horizon)
#         preds.append(pred)
#     elapsed_time = time.time() - start_time
    
#     smapes = eval_forecasts(preds, test_series, kwargs.get("cfg"))
#     return smapes, elapsed_time

# def initialize_model(cfg) -> NBEATSModel:
#     """
#     Initialize the NBEATS model with the configuration, including device settings.

#     Args:
#         cfg (dict): Configuration dictionary.

#     Returns:
#         NBEATSModel: Initialized NBEATS model.
#     """
#     return NBEATSModel(
#         input_chunk_length=cfg["input_chunk_length"],
#         output_chunk_length=cfg["output_chunk_length"],
#         pl_trainer_kwargs=cfg["pl_trainer_kwargs"],
#         batch_size=cfg.get("batch_size", 32)
#     )

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

def load_model(model_path: str) -> NBEATSModel: #removed cfg
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

# helper functions from the official Darts tutorial

# evaluate a set of forecasts for multiple series (in darts format)    
def eval_forecasts(pred_series: List[TimeSeries], 
                   test_series: List[TimeSeries], cfg) -> List[float]:
  
    print('computing sMAPEs...')
    smapes = smape(test_series, pred_series)
    mean, std = np.round(np.mean(smapes),4), np.round(np.std(smapes),4)
    print('Avg sMAPE: %.3f +- %.3f' % (mean, std))
    plt.figure(figsize = (cfg.img_dim1,cfg.img_dim2), dpi=144)
    plt.hist(smapes, bins=50)
    plt.ylabel('Count')
    plt.xlabel('sMAPE')
    plt.show()
    plt.close()
    return smapes


def eval_local_model(train_series: List[TimeSeries], 
                     test_series: List[TimeSeries], 
                     horizon,
                     model_cls, 
                     **kwargs) -> Tuple[List[float], float]:
    preds = []
    start_time = time.time()
    for series in tq.tqdm(train_series):
        model = model_cls(**kwargs)
        model.fit(series)
        pred = model.predict(n=horizon)
        preds.append(pred)
    elapsed_time = time.time() - start_time
    
    smapes = eval_forecasts(preds, test_series)
    return smapes, elapsed_time
