# pipelines/training_pipeline/run_training_pipeline.py
import json
import os
from pipelines.training_pipeline.train          import DistilBertTrainer
from pipelines.training_pipeline.evaluate       import DistilBertEvaluator
from pipelines.training_pipeline.register_model import ModelRegistry


class TrainingPipeline:
    """Orchestrates train → evaluate → register steps."""

    def __init__(self):
        self.trainer    = DistilBertTrainer()
        self.evaluator  = DistilBertEvaluator()
        self.registry   = ModelRegistry()

    def run(self) -> dict:
        print("=" * 60)
        print("  TRAINING PIPELINE STARTED")
        print("=" * 60)

        print("\n[1/3] Fine-tuning DistilBERT...")
        train_metrics = self.trainer.run()

        print("\n[2/3] Running full evaluation on test set...")
        eval_metrics = self.evaluator.run()

        print("\n[3/3] Comparing runs and registering best model...")
        registry_result = self.registry.run(train_metrics)

        os.makedirs("model", exist_ok=True)
        with open("model/best_run.json", "w") as f:
            json.dump(registry_result, f, indent=2)
        print("Saved to model/best_run.json")

        self._print_summary(train_metrics, eval_metrics, registry_result)
        return {
            "train":    train_metrics,
            "eval":     eval_metrics,
            "registry": registry_result
        }

    def _print_summary(self, train, eval_, registry):
        print("\n" + "=" * 60)
        print("  TRAINING PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Train F1   : {train.get('f1_score', 'N/A')}")
        print(f"  Eval F1    : {eval_.get('f1_score', 'N/A')}")
        print(f"  Best model : {registry.get('best_run_name', 'N/A')}")
        print("=" * 60)


if __name__ == "__main__":
    TrainingPipeline().run()
