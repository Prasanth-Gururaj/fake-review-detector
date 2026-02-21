from datasets import load_dataset
import pandas as pd
import os

def ingest_amazon_polarity(n_samples: int = 50000) -> pd.DataFrame:
    print(f'Ingesting {n_samples} samples from Amazon Polarity...')
    dataset = load_dataset('amazon_polarity', split='train')
    dataset = dataset.shuffle(seed=42).select(range(n_samples))
    df = pd.DataFrame({
        'text':    dataset['content'],
        'label':   dataset['label'],
        'is_fake': 0
    })
    print(f'Ingested {len(df)} rows')
    return df