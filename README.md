# Fake Review Detector

Fine-tuned DistilBERT deployed via FastAPI on AWS EC2.
MLflow experiment tracking + Evidently AI drift monitoring.

## Quickstart
```bash
docker-compose up --build
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"text": "Amazing product!!!"}'
```