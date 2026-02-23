# config/config_loader.py
import yaml
import os

# Always resolve relative to THIS file's location — works from anywhere
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_section(section: str) -> dict:
    config = load_config()
    if section not in config:
        raise KeyError(f"Section '{section}' not found in config.yaml")
    return config[section]

def data_config()         -> dict: return get_section("data")
def mlflow_config()       -> dict: return get_section("mlflow")
def baseline_config()     -> dict: return get_section("baseline_model")
def distilbert_config()   -> dict: return get_section("distilbert_model")
def optimization_config() -> dict: return get_section("optimization")
def deployment_config()   -> dict: return get_section("deployment")
def monitoring_config()   -> dict: return get_section("monitoring")
