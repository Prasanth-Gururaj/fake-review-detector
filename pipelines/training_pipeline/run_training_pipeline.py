# pipelines/training_pipeline/run_training_pipeline.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pipelines.training_pipeline.train import train_baseline

def run():
    print("--- TRAINING PIPELINE STARTED ---")
    metrics = train_baseline()
    print("--- TRAINING PIPELINE COMPLETE ---")
    return metrics

if __name__ == "__main__":
    run()
