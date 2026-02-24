# api/serve.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from api.predict import ReviewPredictor, PredictionResult


# ------------------------------------------------------------------ #
# Request / Response Schemas                                          #
# ------------------------------------------------------------------ #

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      example="This product is absolutely amazing!!!")

class BatchReviewRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=100)

class PredictionResponse(BaseModel):
    label:                str
    confidence:           float
    is_fake:              bool
    fake_score:           float
    exclamation_count:    int
    caps_ratio:           float
    word_count:           int
    repetition_score:     float
    generic_phrase_count: int
    unique_word_ratio:    float
    punctuation_density:  float
    explanation:          str

class HealthResponse(BaseModel):
    status:  str
    model:   str
    version: str


# ------------------------------------------------------------------ #
# App Lifespan — loads model once at startup                         #
# ------------------------------------------------------------------ #

predictor: Optional[ReviewPredictor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("🚀 Loading model at startup...")
    predictor = ReviewPredictor()
    predictor.model_loader.load()   # eagerly load — no cold start on first request
    print("✅ Model ready.")
    yield
    print("🛑 Shutting down.")


# ------------------------------------------------------------------ #
# FastAPI App                                                         #
# ------------------------------------------------------------------ #

app = FastAPI(
    title       = "Fake Review Detector API",
    description = "Detects fake reviews using DistilBERT + linguistic feature analysis",
    version     = "1.0.0",
    lifespan    = lifespan
)


# ------------------------------------------------------------------ #
# Endpoints                                                           #
# ------------------------------------------------------------------ #

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Check if the API and model are running."""
    return HealthResponse(
        status  = "healthy",
        model   = "distilbert-base-uncased",
        version = "1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: ReviewRequest):
    """
    Predict whether a single review is fake or genuine.
    Returns model confidence + linguistic feature signals + explanation.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    result = predictor.predict(request.text)
    return PredictionResponse(**result.__dict__)


@app.post("/predict/batch", response_model=list[PredictionResponse], tags=["Prediction"])
def predict_batch(request: BatchReviewRequest):
    """Predict fake/genuine for a batch of reviews (max 100)."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    results = predictor.predict_batch(request.texts)
    return [PredictionResponse(**r.__dict__) for r in results]


@app.get("/features", tags=["Analysis"])
def extract_features(text: str):
    """
    Extract only the linguistic feature signals without running the model.
    Useful for debugging and analysis.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    signals = predictor.extractor.extract_as_dict(text)
    return signals


@app.get("/model/info", tags=["System"])
def model_info():
    """Returns model metadata and configuration."""
    from config.config_loader import deployment_config, distilbert_config
    return {
        "model_config":      distilbert_config(),
        "deployment_config": deployment_config(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.serve:app", host="0.0.0.0", port=8000, reload=True)
