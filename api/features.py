import numpy as np

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