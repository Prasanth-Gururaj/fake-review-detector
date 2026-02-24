
# api/features.py
import re
import string
from dataclasses import dataclass, field
import numpy as np
from typing import Optional


GENERIC_PHRASES = [
    'highly recommend', 'great product', 'best ever', 'love it',
    'five stars', 'amazing', 'perfect', 'best purchase', 'must buy'
]
SPECIFIC_WORDS = [
    'battery', 'screen', 'size', 'quality', 'delivery', 'material',
    'fit', 'texture', 'smell', 'flavor', 'weight', 'color', 'stitching'
]

def extract_fake_features(text: str) -> dict:
    words = text.split()
    return {
        'length':            len(words),
        'exclamation_count': text.count('!'),
        'caps_ratio':        sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'avg_word_len':      float(np.mean([len(w) for w in words])) if words else 0.0,
        'generic_hit':       int(any(p in text.lower() for p in GENERIC_PHRASES)),
        'specific_hit':      int(any(w in text.lower() for w in SPECIFIC_WORDS)),
        'unique_word_ratio': len(set(words)) / max(len(words), 1),
        'sentence_count':    text.count('.') + text.count('!') + text.count('?')
    }

def fake_score(features: dict) -> float:
    signals = [
        features['exclamation_count'] > 3,
        features['length'] < 15,
        features['generic_hit'] == 1 and features['specific_hit'] == 0,
        features['caps_ratio'] > 0.3,
        features['unique_word_ratio'] < 0.5
    ]
    return round(sum(signals) / len(signals), 2)




@dataclass
class FeatureSignals:
    """Container for all extracted feature signals."""
    exclamation_count:    int   = 0
    question_count:       int   = 0
    caps_ratio:           float = 0.0
    word_count:           int   = 0
    char_count:           int   = 0
    unique_word_ratio:    float = 0.0
    avg_word_length:      float = 0.0
    repetition_score:     float = 0.0
    generic_phrase_count: int   = 0
    punctuation_density:  float = 0.0
    fake_score:           float = 0.0


class ExclamationAnalyzer:
    """Detects excessive punctuation abuse."""

    def analyze(self, text: str) -> dict:
        return {
            "exclamation_count": text.count("!"),
            "question_count":    text.count("?"),
        }


class CapsAnalyzer:
    """Detects ALL CAPS abuse — strong fake review signal."""

    def analyze(self, text: str) -> dict:
        words      = text.split()
        if not words:
            return {"caps_ratio": 0.0}
        caps_words = [w for w in words if w.isupper() and len(w) > 1]
        return {"caps_ratio": round(len(caps_words) / len(words), 4)}


class LengthAnalyzer:
    """Analyzes review length — too short or too generic is suspicious."""

    def analyze(self, text: str) -> dict:
        words = text.split()
        chars = len(text.strip())
        avg_word_len = (
            round(sum(len(w) for w in words) / len(words), 4)
            if words else 0.0
        )
        return {
            "word_count":     len(words),
            "char_count":     chars,
            "avg_word_length": avg_word_len,
        }


class RepetitionAnalyzer:
    """Detects repeated words/phrases — a common fake review pattern."""

    def analyze(self, text: str) -> dict:
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return {"unique_word_ratio": 0.0, "repetition_score": 0.0}
        unique_ratio     = round(len(set(words)) / len(words), 4)
        repetition_score = round(1.0 - unique_ratio, 4)
        return {
            "unique_word_ratio": unique_ratio,
            "repetition_score":  repetition_score,
        }


class GenericPhraseDetector:
    """Detects boilerplate phrases common in fake reviews."""

    GENERIC_PHRASES = [
        "best product ever", "highly recommend", "love it",
        "amazing product", "five stars", "must buy",
        "changed my life", "exceeded my expectations",
        "best purchase", "absolutely love", "perfect product",
        "worth every penny", "do not hesitate", "works perfectly",
        "great quality", "fast shipping", "very happy",
    ]

    def analyze(self, text: str) -> dict:
        text_lower = text.lower()
        count      = sum(1 for phrase in self.GENERIC_PHRASES
                         if phrase in text_lower)
        return {"generic_phrase_count": count}


class PunctuationAnalyzer:
    """Measures overall punctuation density."""

    def analyze(self, text: str) -> dict:
        if not text:
            return {"punctuation_density": 0.0}
        punct_count = sum(1 for c in text if c in string.punctuation)
        density     = round(punct_count / len(text), 4)
        return {"punctuation_density": density}


class FakeScoreCalculator:
    """
    Combines all signals into a single fake probability score (0.0 → 1.0).
    Weights are tuned based on known fake review patterns.
    """

    WEIGHTS = {
        "exclamation_count":    0.20,
        "caps_ratio":           0.20,
        "repetition_score":     0.15,
        "generic_phrase_count": 0.20,
        "punctuation_density":  0.10,
        "short_review_penalty": 0.15,
    }

    def calculate(self, signals: dict) -> float:
        exclamation_norm  = min(signals.get("exclamation_count", 0) / 5.0, 1.0)
        caps_norm         = min(signals.get("caps_ratio", 0.0) * 2.0, 1.0)
        repetition_norm   = min(signals.get("repetition_score", 0.0) * 2.0, 1.0)
        generic_norm      = min(signals.get("generic_phrase_count", 0) / 3.0, 1.0)
        punct_norm        = min(signals.get("punctuation_density", 0.0) * 10.0, 1.0)
        short_penalty     = 1.0 if signals.get("word_count", 100) < 10 else 0.0

        score = (
            exclamation_norm  * self.WEIGHTS["exclamation_count"]    +
            caps_norm         * self.WEIGHTS["caps_ratio"]           +
            repetition_norm   * self.WEIGHTS["repetition_score"]     +
            generic_norm      * self.WEIGHTS["generic_phrase_count"] +
            punct_norm        * self.WEIGHTS["punctuation_density"]  +
            short_penalty     * self.WEIGHTS["short_review_penalty"]
        )
        return round(min(score, 1.0), 4)


class FeatureExtractor:
    """
    Orchestrates all analyzers and returns a FeatureSignals dataclass.
    This is the main class called by predict.py on every request.
    """

    def __init__(self):
        self.exclamation_analyzer  = ExclamationAnalyzer()
        self.caps_analyzer         = CapsAnalyzer()
        self.length_analyzer       = LengthAnalyzer()
        self.repetition_analyzer   = RepetitionAnalyzer()
        self.generic_detector      = GenericPhraseDetector()
        self.punctuation_analyzer  = PunctuationAnalyzer()
        self.score_calculator      = FakeScoreCalculator()

    def extract(self, text: str) -> FeatureSignals:
        signals = {}
        signals.update(self.exclamation_analyzer.analyze(text))
        signals.update(self.caps_analyzer.analyze(text))
        signals.update(self.length_analyzer.analyze(text))
        signals.update(self.repetition_analyzer.analyze(text))
        signals.update(self.generic_detector.analyze(text))
        signals.update(self.punctuation_analyzer.analyze(text))
        signals["fake_score"] = self.score_calculator.calculate(signals)
        return FeatureSignals(**signals)

    def extract_as_dict(self, text: str) -> dict:
        signals = self.extract(text)
        return {
            "exclamation_count":    signals.exclamation_count,
            "question_count":       signals.question_count,
            "caps_ratio":           signals.caps_ratio,
            "word_count":           signals.word_count,
            "char_count":           signals.char_count,
            "unique_word_ratio":    signals.unique_word_ratio,
            "avg_word_length":      signals.avg_word_length,
            "repetition_score":     signals.repetition_score,
            "generic_phrase_count": signals.generic_phrase_count,
            "punctuation_density":  signals.punctuation_density,
            "fake_score":           signals.fake_score,
        }


if __name__ == "__main__":
    extractor = FeatureExtractor()

    fake_review = "AMAZING!!! BEST PRODUCT EVER!!! HIGHLY RECOMMEND!!!"
    real_review = "Battery life is decent, around 8 hours on normal use. Screen resolution could be better."

    print("FAKE REVIEW:")
    for k, v in extractor.extract_as_dict(fake_review).items():
        print(f"  {k:<25}: {v}")

    print("\nREAL REVIEW:")
    for k, v in extractor.extract_as_dict(real_review).items():
        print(f"  {k:<25}: {v}")
