# pipelines/monitoring_pipeline/logger.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
from datetime import datetime
from config.config_loader import monitoring_config

def log_prediction(text: str, result: dict):
    cfg      = monitoring_config()
    log_file = cfg["log_file"]
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    entry = {
        "timestamp":  datetime.utcnow().isoformat(),
        "input_text": text,
        **result
    }
    logs = []
    if os.path.exists(log_file):
        with open(log_file) as f:
            try:
                logs = json.load(f)
            except Exception:
                logs = []
    logs.append(entry)
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)
