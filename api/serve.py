from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time, uuid

app = FastAPI(
    title='Fake Review Detector API',
    description='Fine-tuned DistilBERT + ONNX Runtime',
    version='1.0.0'
)

app.add_middleware(CORSMiddleware, allow_origins=['*'])

class ReviewRequest(BaseModel):
    text: str
    user_id: Optional[str] = None

class BatchRequest(BaseModel):
    reviews: List[str]

@app.get('/health')
def health():
    return {'status': 'healthy', 'model': 'distilbert-onnx-int8', 'version': '1.0.0'}

@app.post('/predict')
async def predict(request: ReviewRequest):
    return {'message': 'Model not loaded yet - wire on Day 6', 'text': request.text}

@app.post('/batch')
async def batch_predict(request: BatchRequest):
    return {'message': 'Batch endpoint ready', 'count': len(request.reviews)}

@app.get('/metrics')
def metrics():
    return {'message': 'Metrics endpoint - wire on Day 9'}

@app.get('/drift')
def drift():
    return {'message': 'Drift endpoint - wire on Day 9'}