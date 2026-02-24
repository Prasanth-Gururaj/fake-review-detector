# api/predict.py
import torch
import numpy as np
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from api.features import FeatureExtractor, FeatureSignals
from config.config_loader import deployment_config, distilbert_config


@dataclass
class PredictionResult:
    """Complete prediction response with model output + feature signals."""
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


class ModelLoader:
    """Handles loading and caching the DistilBERT model."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._tokenizer = None
        self._model     = None

    def load(self):
        if self._tokenizer is None or self._model is None:
            print(f"Loading model from {self.model_path}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model     = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._model.eval()
            print("Model loaded ✅")
        return self._tokenizer, self._model

    @property
    def tokenizer(self):
        tokenizer, _ = self.load()
        return tokenizer

    @property
    def model(self):
        _, model = self.load()
        return model


class ExplanationBuilder:
    """Builds a human-readable explanation of why a review is flagged."""

    def build(self, signals: FeatureSignals, is_fake: bool) -> str:
        if not is_fake:
            return "Review appears genuine based on linguistic analysis."

        reasons = []
        if signals.exclamation_count >= 3:
            reasons.append(f"excessive exclamation marks ({signals.exclamation_count})")
        if signals.caps_ratio > 0.3:
            reasons.append(f"high ALL CAPS ratio ({signals.caps_ratio:.0%})")
        if signals.generic_phrase_count > 0:
            reasons.append(f"contains {signals.generic_phrase_count} generic phrase(s)")
        if signals.repetition_score > 0.4:
            reasons.append(f"repetitive language (score: {signals.repetition_score:.2f})")
        if signals.word_count < 10:
            reasons.append("review is very short")

        if reasons:
            return "Flagged due to: " + ", ".join(reasons) + "."
        return "Flagged by model based on overall linguistic patterns."


class ReviewPredictor:
    """
    Main predictor — combines DistilBERT model output with
    feature signals to produce a full PredictionResult.
    Called by serve.py on every /predict request.
    """

    LABEL_MAP = {0: "genuine", 1: "fake"}

    def __init__(self):
        d_cfg              = deployment_config()
        db_cfg             = distilbert_config()
        self.model_loader  = ModelLoader(model_path=d_cfg.get("model_path",
                                                               "model/distilbert-final"))
        self.max_length    = db_cfg.get("max_length", 128)
        self.threshold     = d_cfg.get("fake_threshold", 0.5)
        self.extractor     = FeatureExtractor()
        self.explainer     = ExplanationBuilder()

    def predict(self, text: str) -> PredictionResult:
        # 1. Extract linguistic features
        signals = self.extractor.extract(text)

        # 2. Run model inference
        tokenizer = self.model_loader.tokenizer
        model     = self.model_loader.model

        inputs = tokenizer(
            text,
            return_tensors = "pt",
            truncation     = True,
            padding        = True,
            max_length     = self.max_length
        )
        with torch.no_grad():
            logits = model(**inputs).logits

        probs      = torch.softmax(logits, dim=1).numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = round(float(probs[pred_class]), 4)

        # 3. Combine model + feature signals for final decision
        is_fake = (pred_class == 1) or (signals.fake_score >= self.threshold)
        label   = "fake" if is_fake else "genuine"

        # 4. Build explanation
        explanation = self.explainer.build(signals, is_fake)

        return PredictionResult(
            label                = label,
            confidence           = confidence,
            is_fake              = is_fake,
            fake_score           = signals.fake_score,
            exclamation_count    = signals.exclamation_count,
            caps_ratio           = signals.caps_ratio,
            word_count           = signals.word_count,
            repetition_score     = signals.repetition_score,
            generic_phrase_count = signals.generic_phrase_count,
            unique_word_ratio    = signals.unique_word_ratio,
            punctuation_density  = signals.punctuation_density,
            explanation          = explanation,
        )

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        return [self.predict(text) for text in texts]


if __name__ == "__main__":
    predictor = ReviewPredictor()

    fake_review = "AMAZING!!! BEST PRODUCT EVER!!! HIGHLY RECOMMEND!!!"
    real_review = "Battery life is decent, around 8 hours. Screen could be better."

    for review in [fake_review, real_review]:
        result = predictor.predict(review)
        print(f"\nText      : {review[:60]}...")
        print(f"Label     : {result.label}")
        print(f"Confidence: {result.confidence}")
        print(f"Fake Score: {result.fake_score}")
        print(f"Explanation: {result.explanation}")
