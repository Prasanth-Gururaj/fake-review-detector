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
# setup_project.py
import os

def write_file(path, content):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(content)
        print(f"  Created: {path}")
    else:
        print(f"  Skipped (exists): {path}")

def L(*lines):
    return "\n".join(lines)

# ── pipelines/data_pipeline/ ────────────────────────────────────────────────
write_file("pipelines/__init__.py", "")
write_file("pipelines/data_pipeline/__init__.py", "")

write_file("pipelines/data_pipeline/ingest.py", L(
    "from datasets import load_dataset",
    "import pandas as pd",
    "import os",
    "",
    "def ingest_amazon_polarity(n_samples: int = 50000) -> pd.DataFrame:",
    "    print(f'Ingesting {n_samples} samples from Amazon Polarity...')",
    "    dataset = load_dataset('amazon_polarity', split='train')",
    "    dataset = dataset.shuffle(seed=42).select(range(n_samples))",
    "    df = pd.DataFrame({",
    "        'text':    dataset['content'],",
    "        'label':   dataset['label'],",
    "        'is_fake': 0",
    "    })",
    "    print(f'Ingested {len(df)} rows')",
    "    return df",
))

write_file("pipelines/data_pipeline/preprocess.py", L(
    "import pandas as pd",
    "import re",
    "from sklearn.model_selection import train_test_split",
    "",
    "def clean_text(text: str) -> str:",
    "    text = re.sub(r'<.*?>', '', text)       # remove HTML tags",
    "    text = re.sub(r'\\s+', ' ', text).strip() # normalize whitespace",
    "    return text",
    "",
    "def preprocess(df: pd.DataFrame) -> pd.DataFrame:",
    "    print('Cleaning text...')",
    "    df['text'] = df['text'].apply(clean_text)",
    "    df = df.dropna(subset=['text'])",
    "    df = df[df['text'].str.len() > 10]",
    "    return df",
    "",
    "def split_data(df: pd.DataFrame):",
    "    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])",
    "    print(f'Train: {len(train)} | Test: {len(test)}')",
    "    return train, test",
))

write_file("pipelines/data_pipeline/validate.py", L(
    "import pandas as pd",
    "",
    "def validate(df: pd.DataFrame) -> bool:",
    "    checks = {",
    "        'no_nulls':        df['text'].isnull().sum() == 0,",
    "        'has_both_labels': df['label'].nunique() == 2,",
    "        'min_length':      (df['text'].str.len() > 10).all(),",
    "        'min_rows':        len(df) >= 1000,",
    "    }",
    "    for check, passed in checks.items():",
    "        status = 'PASS' if passed else 'FAIL'",
    "        print(f'  [{status}] {check}')",
    "    all_passed = all(checks.values())",
    "    if not all_passed:",
    "        raise ValueError('Data validation failed. Fix issues before training.')",
    "    print('All data validation checks passed.')",
    "    return True",
))

write_file("pipelines/data_pipeline/run_data_pipeline.py", L(
    "import os, sys",
    "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))",
    "from pipelines.data_pipeline.ingest import ingest_amazon_polarity",
    "from pipelines.data_pipeline.preprocess import preprocess, split_data",
    "from pipelines.data_pipeline.validate import validate",
    "",
    "def run():",
    "    print('--- DATA PIPELINE STARTED ---')",
    "    df = ingest_amazon_polarity(n_samples=50000)",
    "    df = preprocess(df)",
    "    validate(df)",
    "    train_df, test_df = split_data(df)",
    "    os.makedirs('data', exist_ok=True)",
    "    train_df.to_csv('data/train.csv', index=False)",
    "    test_df.to_csv('data/test.csv', index=False)",
    "    df.to_csv('data/reviews.csv', index=False)",
    "    print('--- DATA PIPELINE COMPLETE ---')",
    "    print(f'Files saved: data/train.csv, data/test.csv, data/reviews.csv')",
    "",
    "if __name__ == '__main__':",
    "    run()",
))

# ── pipelines/training_pipeline/ ────────────────────────────────────────────
write_file("pipelines/training_pipeline/__init__.py", "")

write_file("pipelines/training_pipeline/train.py", L(
    "# Day 3 - Fine-tune DistilBERT",
    "# Full implementation on Day 3",
    "import mlflow",
    "",
    "def train_model(train_path: str, test_path: str):",
    "    print('Training pipeline - implement on Day 3')",
    "    raise NotImplementedError('Implement on Day 3')",
))

write_file("pipelines/training_pipeline/evaluate.py", L(
    "# Day 3 - Evaluate model and log metrics to MLflow",
    "import mlflow",
    "from sklearn.metrics import f1_score, accuracy_score, classification_report",
    "",
    "def evaluate(model, X_test, y_test, run_name: str = 'evaluation'):",
    "    print('Evaluation pipeline - implement on Day 3')",
    "    raise NotImplementedError('Implement on Day 3')",
))

write_file("pipelines/training_pipeline/register_model.py", L(
    "# Day 3 - Promote best MLflow run to model registry",
    "import mlflow",
    "",
    "def register_best_model(experiment_name: str, metric: str = 'f1_score'):",
    "    print('Model registration - implement on Day 3')",
    "    raise NotImplementedError('Implement on Day 3')",
))

write_file("pipelines/training_pipeline/run_training_pipeline.py", L(
    "import os, sys",
    "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))",
    "",
    "def run():",
    "    print('--- TRAINING PIPELINE STARTED ---')",
    "    print('Implement on Day 3')",
    "    print('--- TRAINING PIPELINE COMPLETE ---')",
    "",
    "if __name__ == '__main__':",
    "    run()",
))

# ── pipelines/deployment_pipeline/ ──────────────────────────────────────────
write_file("pipelines/deployment_pipeline/__init__.py", "")

write_file("pipelines/deployment_pipeline/optimize.py", L(
    "# Day 4 - Quantize model + export to ONNX",
    "import torch",
    "",
    "def quantize_and_export(model_path: str, output_path: str = 'model/model.onnx'):",
    "    print('Optimization pipeline - implement on Day 4')",
    "    raise NotImplementedError('Implement on Day 4')",
))

write_file("pipelines/deployment_pipeline/predict.py", L(
    "# Day 6 - ONNX inference",
    "",
    "def classify_review(text: str) -> dict:",
    "    raise NotImplementedError('Implement on Day 6')",
    "",
    "def classify_batch(texts: list) -> list:",
    "    raise NotImplementedError('Implement on Day 6')",
))

write_file("pipelines/deployment_pipeline/run_deployment_pipeline.py", L(
    "import os, sys",
    "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))",
    "",
    "def run():",
    "    print('--- DEPLOYMENT PIPELINE STARTED ---')",
    "    print('Step 1: Optimize model (Day 4)')",
    "    print('Step 2: Build Docker image (Day 7)')",
    "    print('Step 3: Deploy to EC2 (Day 8)')",
    "    print('--- DEPLOYMENT PIPELINE COMPLETE ---')",
    "",
    "if __name__ == '__main__':",
    "    run()",
))

# ── pipelines/monitoring_pipeline/ ──────────────────────────────────────────
write_file("pipelines/monitoring_pipeline/__init__.py", "")

write_file("pipelines/monitoring_pipeline/logger.py", L(
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
))

write_file("pipelines/monitoring_pipeline/drift_monitor.py", L(
    "# Day 9 - Evidently AI drift detection",
    "",
    "def run_drift_check() -> bool:",
    "    print('Drift monitor - implement on Day 9')",
    "    return False",
))

write_file("pipelines/monitoring_pipeline/performance_monitor.py", L(
    "# Day 9 - Track model performance post-deployment",
    "import json, os",
    "",
    "def get_performance_summary() -> dict:",
    "    log_file = 'monitoring/prediction_log.json'",
    "    if not os.path.exists(log_file):",
    "        return {'message': 'No predictions logged yet'}",
    "    with open(log_file) as f:",
    "        logs = json.load(f)",
    "    if not logs:",
    "        return {'message': 'No predictions logged yet'}",
    "    avg_latency    = sum(l.get('latency_ms', 0) for l in logs) / len(logs)",
    "    avg_confidence = sum(l.get('confidence', 0) for l in logs) / len(logs)",
    "    fake_rate      = sum(1 for l in logs if l.get('fake_score', 0) > 0.5) / len(logs)",
    "    return {",
    "        'total_predictions': len(logs),",
    "        'avg_latency_ms':    round(avg_latency, 2),",
    "        'avg_confidence':    round(avg_confidence, 3),",
    "        'fake_detection_rate': round(fake_rate, 3)",
    "    }",
))

write_file("pipelines/monitoring_pipeline/run_monitoring_pipeline.py", L(
    "import os, sys",
    "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))",
    "from pipelines.monitoring_pipeline.drift_monitor import run_drift_check",
    "from pipelines.monitoring_pipeline.performance_monitor import get_performance_summary",
    "",
    "def run():",
    "    print('--- MONITORING PIPELINE STARTED ---')",
    "    summary = get_performance_summary()",
    "    print(f'Performance Summary: {summary}')",
    "    drift_detected = run_drift_check()",
    "    print(f'Drift Detected: {drift_detected}')",
    "    if drift_detected:",
    "        print('ACTION: Trigger retraining pipeline')",
    "    print('--- MONITORING PIPELINE COMPLETE ---')",
    "",
    "if __name__ == '__main__':",
    "    run()",
))

# ── api/ ─────────────────────────────────────────────────────────────────────
write_file("api/__init__.py", "")

write_file("api/features.py", L(
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
))

write_file("api/serve.py", L(
    "from fastapi import FastAPI, BackgroundTasks",
    "from fastapi.middleware.cors import CORSMiddleware",
    "from pydantic import BaseModel",
    "from typing import List, Optional",
    "import time, uuid",
    "from pipelines.monitoring_pipeline.logger import log_prediction",
    "from pipelines.monitoring_pipeline.performance_monitor import get_performance_summary",
    "from pipelines.monitoring_pipeline.drift_monitor import run_drift_check",
    "",
    "app = FastAPI(",
    "    title='Fake Review Detector API',",
    "    description='Fine-tuned DistilBERT + ONNX Runtime | MLflow Tracked | Evidently Monitored',",
    "    version='1.0.0'",
    ")",
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
    "async def predict(request: ReviewRequest, background_tasks: BackgroundTasks):",
    "    start = time.time()",
    "    # Wire to ONNX model on Day 6",
    "    result = {'sentiment': 'PENDING', 'confidence': 0.0, 'fake_score': 0.0, 'verdict': 'pending'}",
    "    result['latency_ms'] = round((time.time() - start) * 1000, 2)",
    "    result['request_id'] = str(uuid.uuid4())",
    "    background_tasks.add_task(log_prediction, request.text, result)",
    "    return result",
    "",
    "@app.post('/batch')",
    "async def batch_predict(request: BatchRequest):",
    "    return {'message': 'Batch endpoint - wire on Day 6', 'count': len(request.reviews)}",
    "",
    "@app.get('/metrics')",
    "def metrics():",
    "    return get_performance_summary()",
    "",
    "@app.get('/drift')",
    "def drift():",
    "    detected = run_drift_check()",
    "    return {'drift_detected': detected, 'action': 'Retraining triggered' if detected else 'Model stable'}",
))

# ── tests/ ───────────────────────────────────────────────────────────────────
write_file("tests/__init__.py", "")

write_file("tests/test_data_pipeline.py", L(
    "import os, sys",
    "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))",
    "import pandas as pd",
    "from pipelines.data_pipeline.preprocess import clean_text, preprocess, split_data",
    "from pipelines.data_pipeline.validate import validate",
    "",
    "def test_clean_text_removes_html():",
    "    assert clean_text('<b>Hello</b>') == 'Hello'",
    "",
    "def test_clean_text_normalizes_whitespace():",
    "    assert clean_text('Hello   world') == 'Hello world'",
    "",
    "def test_validate_passes_good_data():",
    "    df = pd.DataFrame({'text': ['good review text'] * 1000 + ['bad review'] * 1000, 'label': [1]*1000 + [0]*1000})",
    "    assert validate(df) == True",
    "",
    "def test_split_ratio():",
    "    df = pd.DataFrame({'text': ['review'] * 1000, 'label': [1]*500 + [0]*500})",
    "    train, test = split_data(df)",
    "    assert len(test) == 200",
    "    assert len(train) == 800",
))

write_file("tests/test_api.py", L(
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
    "def test_predict_returns_200():",
    "    response = client.post('/predict', json={'text': 'This is a test review'})",
    "    assert response.status_code == 200",
    "",
    "def test_metrics_endpoint():",
    "    response = client.get('/metrics')",
    "    assert response.status_code == 200",
    "",
    "def test_drift_endpoint():",
    "    response = client.get('/drift')",
    "    assert response.status_code == 200",
    "    assert 'drift_detected' in response.json()",
))

write_file("tests/test_features.py", L(
    "from api.features import extract_fake_features, fake_score",
    "",
    "def test_fake_review_scores_high():",
    "    fake = 'AMAZING!!! BEST PRODUCT EVER!!! HIGHLY RECOMMEND!!!'",
    "    assert fake_score(extract_fake_features(fake)) >= 0.5",
    "",
    "def test_genuine_review_scores_low():",
    "    genuine = 'Battery life lasts about 8 hours. Screen resolution is decent but not great outdoors.'",
    "    assert fake_score(extract_fake_features(genuine)) < 0.5",
))

# ── Root config files ────────────────────────────────────────────────────────
write_file("Dockerfile", L(
    "FROM python:3.11-slim",
    "WORKDIR /app",
    "COPY requirements.txt .",
    "RUN pip install --no-cache-dir -r requirements.txt",
    "COPY . .",
    "EXPOSE 8000",
    'CMD ["uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]',
))

write_file("docker-compose.yml", L(
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
))

write_file(".env", L(
    "MLFLOW_TRACKING_URI=http://localhost:5000",
    "MODEL_PATH=model/model.onnx",
    "LOG_FILE=monitoring/prediction_log.json",
))

write_file(".gitignore", L(
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
))

write_file("README.md", L(
    "# Fake Review Detector",
    "",
    "Fine-tuned DistilBERT deployed via FastAPI on AWS EC2.",
    "4-stage ML pipeline: Data -> Training -> Deployment -> Monitoring",
    "",
    "## Pipelines",
    "- `pipelines/data_pipeline/`       - Ingest, preprocess, validate",
    "- `pipelines/training_pipeline/`   - Train, evaluate, register model",
    "- `pipelines/deployment_pipeline/` - Optimize (ONNX), serve (FastAPI)",
    "- `pipelines/monitoring_pipeline/` - Drift detection, performance tracking",
    "",
    "## Run Each Pipeline",
    "```bash",
    "python pipelines/data_pipeline/run_data_pipeline.py",
    "python pipelines/training_pipeline/run_training_pipeline.py",
    "python pipelines/deployment_pipeline/run_deployment_pipeline.py",
    "python pipelines/monitoring_pipeline/run_monitoring_pipeline.py",
    "```",
))

write_file("data/.gitkeep", "")
write_file("model/.gitkeep", "")
write_file("monitoring/.gitkeep", "")

print("\nPipeline structure created successfully!")
print("\nNext steps:")
print("  1. python pipelines/data_pipeline/run_data_pipeline.py")
print("  2. pytest tests/")
print("  3. uvicorn api.serve:app --reload")
