import json, os
from datetime import datetime

LOG_FILE = 'monitoring/prediction_log.json'

def log_prediction(text: str, result: dict):
    os.makedirs('monitoring', exist_ok=True)
    entry = {'timestamp': datetime.utcnow().isoformat(), 'input_text': text, **result}
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            try:
                logs = json.load(f)
            except Exception:
                logs = []
    logs.append(entry)
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)