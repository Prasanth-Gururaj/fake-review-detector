# pipelines/training_pipeline/evaluate.py
import os
import pandas as pd
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)
from config.config_loader import data_config, distilbert_config


def evaluate_distilbert(model_path: str = "model/distilbert-final"):
    d_cfg  = data_config()
    db_cfg = distilbert_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    test_df = pd.read_csv(d_cfg["test_path"])
    texts   = test_df["text"].astype(str).tolist()
    labels  = test_df["label"].tolist()

    all_preds = []
    batch_size = 32

    print(f"Running inference on {len(texts)} samples...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=db_cfg["max_length"]
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        preds = np.argmax(outputs.logits.cpu().numpy(), axis=1)
        all_preds.extend(preds.tolist())
        if i % 2000 == 0:
            print(f"  {i}/{len(texts)} done...")

    metrics = {
        "accuracy":  round(accuracy_score(labels, all_preds), 4),
        "f1_score":  round(f1_score(labels, all_preds), 4),
        "precision": round(precision_score(labels, all_preds), 4),
        "recall":    round(recall_score(labels, all_preds), 4),
    }

    cm = confusion_matrix(labels, all_preds)

    print("\n" + "="*50)
    print("  FULL TEST SET EVALUATION")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print("\nClassification Report:")
    print(classification_report(labels, all_preds,
          target_names=["Negative", "Positive"]))

    os.makedirs("model", exist_ok=True)
    with open("model/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved to model/eval_metrics.json")

    return metrics


if __name__ == "__main__":
    evaluate_distilbert()
