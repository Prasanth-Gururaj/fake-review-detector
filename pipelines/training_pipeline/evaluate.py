# pipelines/training_pipeline/evaluate.py
import os
import json
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, classification_report, confusion_matrix)
from config.config_loader import data_config, distilbert_config


class DistilBertEvaluator:
    """Runs full test set evaluation on the saved DistilBERT model."""

    def __init__(self, model_path: str = "model/distilbert-final", batch_size: int = 32):
        self.model_path = model_path
        self.batch_size = batch_size
        self.d_cfg      = data_config()
        self.db_cfg     = distilbert_config()
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model     = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def _run_inference(self, tokenizer, model, texts: list) -> list:
        all_preds = []
        print(f"Running inference on {len(texts)} samples...")
        for i in range(0, len(texts), self.batch_size):
            batch  = texts[i:i + self.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                               padding=True, max_length=self.db_cfg["max_length"])
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            preds = np.argmax(outputs.logits.cpu().numpy(), axis=1)
            all_preds.extend(preds.tolist())
            if i % 2000 == 0:
                print(f"  {i}/{len(texts)} done...")
        return all_preds

    def _compute_metrics(self, labels, preds) -> dict:
        return {
            "accuracy":  round(accuracy_score(labels, preds), 4),
            "f1_score":  round(f1_score(labels, preds), 4),
            "precision": round(precision_score(labels, preds), 4),
            "recall":    round(recall_score(labels, preds), 4),
        }

    def _print_results(self, metrics, labels, preds):
        cm = confusion_matrix(labels, preds)
        print("\n" + "="*50)
        print("  FULL TEST SET EVALUATION")
        print("="*50)
        for k, v in metrics.items():
            print(f"  {k:<22}: {v}")
        print(f"\nConfusion Matrix:")
        print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
        print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=["Negative", "Positive"]))

    def _save_metrics(self, metrics: dict):
        os.makedirs("model", exist_ok=True)
        with open("model/eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("Saved to model/eval_metrics.json")

    def run(self) -> dict:
        print(f"Evaluating on: {self.device}")
        tokenizer, model = self._load_model()

        test_df = pd.read_csv(self.d_cfg["test_path"])
        texts   = test_df["text"].astype(str).tolist()
        labels  = test_df["label"].tolist()

        preds   = self._run_inference(tokenizer, model, texts)
        metrics = self._compute_metrics(labels, preds)

        self._print_results(metrics, labels, preds)
        self._save_metrics(metrics)
        return metrics


if __name__ == "__main__":
    DistilBertEvaluator().run()
