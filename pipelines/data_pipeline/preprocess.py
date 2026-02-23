# pipelines/data_pipeline/preprocess.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from config.config_loader import data_config

def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning text...")
    df["text"] = df["text"].apply(clean_text)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() > 10]
    return df

def split_data(df: pd.DataFrame):
    cfg = data_config()
    train, test = train_test_split(
        df,
        test_size=cfg["test_size"],
        random_state=cfg["random_seed"],
        stratify=df["label"]
    )
    print(f"Train: {len(train)} | Test: {len(test)}")
    return train, test
