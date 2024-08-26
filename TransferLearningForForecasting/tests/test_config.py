import pytest
import yaml

@pytest.fixture
def load_config():
    """Fixture to load the config file."""
    with open("TransferLearningForForecasting/configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

def test_forecast_length_less_than_horizon(load_config):
    """Test that the forecast length is smaller than the horizon."""
    model_config = load_config["model"]
    assert model_config["forecast_length"] < model_config["horizon"], (
        "forecast_length should be smaller than horizon"
    )

def test_mandatory_paths(load_config):
    """Test that essential paths are specified in the config."""
    data_config = load_config["data"]
    model_config = load_config["model"]
    output_config = load_config["output"]

    assert data_config.get("path"), "Data path must be specified in the config."
    assert model_config.get("path"), "Model path must be specified in the config."
    if load_config.get("output"):
        assert output_config.get("prediction_save_path"), "Prediction save path must be specified if output is configured."

def test_minimum_series_size(load_config):
    """Test that min_size is a reasonable positive integer."""
    min_size = load_config["model"]["min_size"]
    assert isinstance(min_size, int) and min_size > 0, "min_size must be a positive integer."

def test_pl_trainer_kwargs_validity(load_config):
    """Test that pl_trainer_kwargs contains valid settings."""
    trainer_kwargs = load_config["model"].get("pl_trainer_kwargs", {})
    assert isinstance(trainer_kwargs, dict), "pl_trainer_kwargs should be a dictionary."

    accelerator = trainer_kwargs.get("accelerator")
    assert accelerator in ["cpu", "gpu", "tpu", "mps"], "Invalid accelerator specified in pl_trainer_kwargs."

def test_output_chunk_length_not_greater_than_input(load_config):
    """Test that output_chunk_length is not greater than input_chunk_length."""
    input_chunk_length = load_config["model"]["input_chunk_length"]
    output_chunk_length = load_config["model"]["output_chunk_length"]
    assert output_chunk_length <= input_chunk_length, (
        "output_chunk_length cannot be greater than input_chunk_length."
    )