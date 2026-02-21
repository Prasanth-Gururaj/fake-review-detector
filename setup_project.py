# setup_project.py
import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(content)
        print(f"  ✅ Created: {path}")
    else:
        print(f"  ⚠️  Skipped (exists): {path}")


# ── data/prepare_data.py ────────────────────────────────────────────────────
write_file("data/prepare_data.py", "\n".join([
    "from datasets import load_dataset",
    "import pandas as pd",
    "import os",
    "",
    "os.makedirs('data', exist_ok=True)",
    "",
    "print('Loading Amazon Polarity dataset (50,000 samples)...')",
    "amazon = load_dataset('amazon_polarity', split='train').shuffle(seed=42).select(range(50000))",
    "",
    "df = pd.DataFrame({",
    "    'text':    amazon['content'],",
    "    'label':   amazon['label'],",
    "    'is_fake': 0",
    "})",
    "",
    "df.to_csv('data/reviews.csv', index=False)",
    "print(f'Saved {len(df)} reviews to data/reviews.csv')",
    "print(df.head())",
    "print(df['label'].value_counts())",
]))

# ── api/__init__.py ─────────────────────────────────────────────────────────
write_file("api/__init__.py", "")

# ── api/serve.py ────────────────────────────────────────────────────────────
write_file("api/serve.py", "\n".join([
    "from fastapi import FastAPI, BackgroundTasks",
    "from fastapi.middleware.cors import CORSMiddleware",
    "from pydantic import BaseModel",
    "from typing import List, Optional",
    "import time, uuid",
    "",
    "app = FastAPI(",
    "    title='Fake Review Detector API',",
    "    description='Fine-tuned DistilBERT + ONNX Runtime',",
    "    version='1.0.0'",
    ")",
    "",
    "app.add_middleware(CORSMiddleware, allow_origins=['*'])",
    "",
    "class ReviewRequest(BaseModel):",
    "    text: str",
    "    user_id: Optional[str] = None",
    "",
    "class BatchRequest(BaseModel):",
    "    reviews: List[str]",
    "",
    "@app.get('/health')",
    "def health():",
    "    return {'status': 'healthy', 'model': 'distilbert-onnx-int8', 'version': '1.0.0'}",
    "",
    "@app.post('/predict')",
    "async def predict(request: ReviewRequest):",
    "    return {'message': 'Model not loaded yet - wire on Day 6', 'text': request.text}",
    "",
    "@app.post('/batch')",
    "async def batch_predict(request: BatchRequest):",
    "    return {'message': 'Batch endpoint ready', 'count': len(request.reviews)}",
    "",
    "@app.get('/metrics')",
    "def metrics():",
    "    return {'message': 'Metrics endpoint - wire on Day 9'}",
    "",
    "@app.get('/drift')",
    "def drift():",
    "    return {'message': 'Drift endpoint - wire on Day 9'}",
]))

# ── api/predict.py ──────────────────────────────────────────────────────────
write_file("api/predict.py", "\n".join([
    "# Day 6 - Wire ONNX model here",
    "",
    "def classify_review(text: str) -> dict:",
    "    raise NotImplementedError('Wire ONNX model on Day 6')",
    "",
    "def classify_batch(texts: list) -> list:",
    "    raise NotImplementedError('Wire ONNX model on Day 6')",
]))

# ── api/features.py ─────────────────────────────────────────────────────────
write_file("api/features.py", "\n".join([
    "import numpy as np",
    "",
    "GENERIC_PHRASES = [",
    "    'highly recommend', 'great product', 'best ever', 'love it',",
    "    'five stars', 'amazing', 'perfect', 'best purchase', 'must buy'",
    "]",
    "SPECIFIC_WORDS = [",
    "    'battery', 'screen', 'size', 'quality', 'delivery', 'material',",
    "    'fit', 'texture', 'smell', 'flavor', 'weight', 'color', 'stitching'",
    "]",
    "",
    "def extract_fake_features(text: str) -> dict:",
    "    words = text.split()",
    "    return {",
    "        'length':            len(words),",
    "        'exclamation_count': text.count('!'),",
    "        'caps_ratio':        sum(1 for c in text if c.isupper()) / max(len(text), 1),",
    "        'avg_word_len':      float(np.mean([len(w) for w in words])) if words else 0.0,",
    "        'generic_hit':       int(any(p in text.lower() for p in GENERIC_PHRASES)),",
    "        'specific_hit':      int(any(w in text.lower() for w in SPECIFIC_WORDS)),",
    "        'unique_word_ratio': len(set(words)) / max(len(words), 1),",
    "        'sentence_count':    text.count('.') + text.count('!') + text.count('?')",
    "    }",
    "",
    "def fake_score(features: dict) -> float:",
    "    signals = [",
    "        features['exclamation_count'] > 3,",
    "        features['length'] < 15,",
    "        features['generic_hit'] == 1 and features['specific_hit'] == 0,",
    "        features['caps_ratio'] > 0.3,",
    "        features['unique_word_ratio'] < 0.5",
    "    ]",
    "    return round(sum(signals) / len(signals), 2)",
]))

# ── model/ ───────────────────────────────────────────────────────────────────
write_file("model/__init__.py", "")
write_file("model/train.py",    "# Day 3 - Fine-tune DistilBERT\nprint('Implement on Day 3')")
write_file("model/optimize.py", "# Day 4 - Quantize + ONNX export\nprint('Implement on Day 4')")

# ── monitoring/ ──────────────────────────────────────────────────────────────
write_file("monitoring/__init__.py", "")

write_file("monitoring/logger.py", "\n".join([
    "import json, os",
    "from datetime import datetime",
    "",
    "LOG_FILE = 'monitoring/prediction_log.json'",
    "",
    "def log_prediction(text: str, result: dict):",
    "    os.makedirs('monitoring', exist_ok=True)",
    "    entry = {'timestamp': datetime.utcnow().isoformat(), 'input_text': text, **result}",
    "    logs = []",
    "    if os.path.exists(LOG_FILE):",
    "        with open(LOG_FILE) as f:",
    "            try:",
    "                logs = json.load(f)",
    "            except Exception:",
    "                logs = []",
    "    logs.append(entry)",
    "    with open(LOG_FILE, 'w') as f:",
    "        json.dump(logs, f, indent=2)",
]))

write_file("monitoring/drift_monitor.py", "\n".join([
    "# Day 9 - Evidently drift detection",
    "",
    "def run_drift_check() -> bool:",
    "    print('Drift monitor - implement on Day 9')",
    "    return False",
]))

# ── tests/ ───────────────────────────────────────────────────────────────────
write_file("tests/__init__.py", "")

write_file("tests/test_api.py", "\n".join([
    "from fastapi.testclient import TestClient",
    "from api.serve import app",
    "",
    "client = TestClient(app)",
    "",
    "def test_health():",
    "    response = client.get('/health')",
    "    assert response.status_code == 200",
    "    assert response.json()['status'] == 'healthy'",
    "",
    "def test_predict_endpoint_exists():",
    "    response = client.post('/predict', json={'text': 'This is a test review'})",
    "    assert response.status_code == 200",
]))

write_file("tests/test_features.py", "\n".join([
    "from api.features import extract_fake_features, fake_score",
    "",
    "def test_fake_review_scores_high():",
    "    fake = 'AMAZING!!! BEST PRODUCT EVER!!! HIGHLY RECOMMEND!!!'",
    "    features = extract_fake_features(fake)",
    "    score = fake_score(features)",
    "    assert score >= 0.5, f'Expected >= 0.5, got {score}'",
    "",
    "def test_genuine_review_scores_low():",
    "    genuine = 'Battery life lasts about 8 hours. Screen resolution is decent but not great outdoors.'",
    "    features = extract_fake_features(genuine)",
    "    score = fake_score(features)",
    "    assert score < 0.5, f'Expected < 0.5, got {score}'",
]))

# ── Root config files ────────────────────────────────────────────────────────
write_file("Dockerfile", "\n".join([
    "FROM python:3.11-slim",
    "WORKDIR /app",
    "COPY requirements.txt .",
    "RUN pip install --no-cache-dir -r requirements.txt",
    "COPY . .",
    "EXPOSE 8000",
    'CMD ["uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]',
]))

write_file("docker-compose.yml", "\n".join([
    "version: '3.8'",
    "services:",
    "  api:",
    "    build: .",
    "    ports:",
    "      - '8000:8000'",
    "    volumes:",
    "      - ./monitoring:/app/monitoring",
    "      - ./model:/app/model",
    "    environment:",
    "      - MLFLOW_TRACKING_URI=http://mlflow:5000",
    "  mlflow:",
    "    image: ghcr.io/mlflow/mlflow",
    "    ports:",
    "      - '5000:5000'",
    "    command: mlflow server --host 0.0.0.0 --port 5000",
]))

write_file(".env", "\n".join([
    "MLFLOW_TRACKING_URI=http://localhost:5000",
    "MODEL_PATH=model/model.onnx",
    "LOG_FILE=monitoring/prediction_log.json",
]))

write_file(".gitignore", "\n".join([
    "__pycache__/",
    "*.pyc",
    ".env",
    "data/*.csv",
    "model/*.onnx",
    "model/*.pt",
    "monitoring/prediction_log.json",
    "mlruns/",
    ".DS_Store",
    "venv/",
    "fake-review-env/",
]))

write_file("README.md", "\n".join([
    "# Fake Review Detector",
    "",
    "Fine-tuned DistilBERT deployed via FastAPI on AWS EC2.",
    "MLflow experiment tracking + Evidently AI drift monitoring.",
    "",
    "## Quickstart",
    "```bash",
    "docker-compose up --build",
    "curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{\"text\": \"Amazing product!!!\"}'",
    "```",
]))
