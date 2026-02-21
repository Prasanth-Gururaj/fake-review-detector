# Day 9 - Track model performance post-deployment
import json, os

def get_performance_summary() -> dict:
    log_file = 'monitoring/prediction_log.json'
    if not os.path.exists(log_file):
        return {'message': 'No predictions logged yet'}
    with open(log_file) as f:
        logs = json.load(f)
    if not logs:
        return {'message': 'No predictions logged yet'}
    avg_latency    = sum(l.get('latency_ms', 0) for l in logs) / len(logs)
    avg_confidence = sum(l.get('confidence', 0) for l in logs) / len(logs)
    fake_rate      = sum(1 for l in logs if l.get('fake_score', 0) > 0.5) / len(logs)
    return {
        'total_predictions': len(logs),
        'avg_latency_ms':    round(avg_latency, 2),
        'avg_confidence':    round(avg_confidence, 3),
        'fake_detection_rate': round(fake_rate, 3)
    }