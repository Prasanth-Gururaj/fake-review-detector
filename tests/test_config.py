# tests/test_config.py
from config.config_loader import (load_config, data_config, mlflow_config,
                                   baseline_config, distilbert_config,
                                   optimization_config, deployment_config,
                                   monitoring_config)

def test_config_loads():
    config = load_config()
    assert config is not None

def test_all_sections_exist():
    config = load_config()
    for section in ["data", "mlflow", "baseline_model", "distilbert_model",
                    "optimization", "deployment", "monitoring"]:
        assert section in config, f"Missing section: {section}"

def test_data_config_has_required_keys():
    cfg = data_config()
    for key in ["train_path", "test_path", "n_samples", "test_size", "random_seed"]:
        assert key in cfg

def test_distilbert_config_has_required_keys():
    cfg = distilbert_config()
    for key in ["model_name", "learning_rate", "epochs", "batch_size", "max_length"]:
        assert key in cfg

def test_monitoring_thresholds_are_valid():
    cfg = monitoring_config()
    assert 0 < cfg["drift_threshold"] < 1
    assert 0 < cfg["fake_score_threshold"] < 1
    assert 0 < cfg["confidence_alert_threshold"] < 1
