# pipelines/monitoring_pipeline/performance_monitor.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
from config.config_loader import monitoring_config

def get_performance_summary() -> dict:
    cfg      = monitoring_config()
    log_file = cfg["log_file"]
    if not os.path.exists(log_file):
        return {"message": "No predictions logged yet"}
    with open(log_file) as f:
        logs = json.load(f)
    if not logs:
        return {"message": "No predictions logged yet"}
    avg_latency    = sum(l.get("latency_ms", 0)   for l in logs) / len(logs)
    avg_confidence = sum(l.get("confidence", 0)   for l in logs) / len(logs)
    fake_rate      = sum(1 for l in logs if l.get("fake_score", 0) >
                        cfg["fake_score_threshold"]) / len(logs)
    alert = avg_confidence < cfg["confidence_alert_threshold"]
    return {
        "total_predictions":   len(logs),
        "avg_latency_ms":      round(avg_latency, 2),
        "avg_confidence":      round(avg_confidence, 3),
        "fake_detection_rate": round(fake_rate, 3),
        "confidence_alert":    alert
    }
