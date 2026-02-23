# pipelines/data_pipeline/ingest.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from datasets import load_dataset
import pandas as pd
from config.config_loader import data_config

def ingest_amazon_polarity() -> pd.DataFrame:
    cfg = data_config()
    n   = cfg["n_samples"]
    print(f"Ingesting {n} samples from Amazon Polarity...")
    dataset = load_dataset("amazon_polarity", split="train")
    dataset = dataset.shuffle(seed=cfg["random_seed"]).select(range(n))
    df = pd.DataFrame({
        "text":    dataset["content"],
        "label":   dataset["label"],
        "is_fake": 0
    })
    print(f"Ingested {len(df)} rows")
    return df
