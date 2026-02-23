# pipelines/data_pipeline/run_data_pipeline.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pipelines.data_pipeline.ingest     import ingest_amazon_polarity
from pipelines.data_pipeline.preprocess import preprocess, split_data
from pipelines.data_pipeline.validate   import validate
from config.config_loader               import data_config

def run():
    cfg = data_config()
    print("--- DATA PIPELINE STARTED ---")
    df       = ingest_amazon_polarity()
    df       = preprocess(df)
    validate(df)
    train_df, test_df = split_data(df)
    os.makedirs("data", exist_ok=True)
    train_df.to_csv(cfg["train_path"],   index=False)
    test_df.to_csv(cfg["test_path"],     index=False)
    df.to_csv(cfg["reviews_path"],       index=False)
    print("--- DATA PIPELINE COMPLETE ---")
    print(f"Saved: {cfg['train_path']}, {cfg['test_path']}, {cfg['reviews_path']}")

if __name__ == "__main__":
    run()
