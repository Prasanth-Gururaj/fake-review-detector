# pipelines/training_pipeline/run_training_pipeline.py
import json
import os
from pipelines.training_pipeline.train          import train_distilbert
from pipelines.training_pipeline.evaluate       import evaluate_distilbert   
from pipelines.training_pipeline.register_model import register_best_model


def run():
    print("=" * 60)
    print("  TRAINING PIPELINE STARTED")
    print("=" * 60)

    print("\n[1/3] Fine-tuning DistilBERT...")
    train_metrics = train_distilbert()

    print("\n[2/3] Running full evaluation on test set...")
    eval_metrics = evaluate_distilbert()

    print("\n[3/3] Comparing runs and registering best model...")
    registry_result = register_best_model(train_metrics)

    os.makedirs("model", exist_ok=True)
    with open("model/best_run.json", "w") as f:
        json.dump(registry_result, f, indent=2)
    print("Saved to model/best_run.json")

    print("\n" + "=" * 60)
    print("  TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Train F1   : {train_metrics.get('f1_score', 'N/A')}")
    print(f"  Eval F1    : {eval_metrics.get('f1_score', 'N/A')}")
    print(f"  Best model : {registry_result.get('best_run_name', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    run()
