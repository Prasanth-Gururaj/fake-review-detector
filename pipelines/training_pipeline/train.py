# pipelines/training_pipeline/train.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model        import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline            import Pipeline
from sklearn.metrics             import (f1_score, accuracy_score,
                                         precision_score, recall_score,
                                         classification_report)
import time
from config.config_loader import data_config, mlflow_config, baseline_config

def train_baseline():
    d_cfg  = data_config()
    ml_cfg = mlflow_config()
    b_cfg  = baseline_config()

    mlflow.set_tracking_uri(ml_cfg["tracking_uri"])
    mlflow.set_experiment(ml_cfg["experiment_name"])

    print("Loading data...")
    train_df = pd.read_csv(d_cfg["train_path"])
    test_df  = pd.read_csv(d_cfg["test_path"])
    X_train, y_train = train_df["text"].astype(str), train_df["label"]
    X_test,  y_test  = test_df["text"].astype(str),  test_df["label"]
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    with mlflow.start_run(run_name="baseline-tfidf-logistic-regression"):

        mlflow.log_params(b_cfg)
        mlflow.log_params({
            "train_samples": len(X_train),
            "test_samples":  len(X_test)
        })

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=b_cfg["max_features"],
                ngram_range=tuple(b_cfg["ngram_range"]),
                stop_words="english"
            )),
            ("clf", LogisticRegression(
                C=b_cfg["C"],
                max_iter=b_cfg["max_iter"],
                random_state=d_cfg["random_seed"]
            ))
        ])

        print("Training...")
        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = round(time.time() - start, 2)

        preds = pipeline.predict(X_test)
        metrics = {
            "accuracy":       round(accuracy_score(y_test, preds), 4),
            "f1_score":       round(f1_score(y_test, preds), 4),
            "precision":      round(precision_score(y_test, preds), 4),
            "recall":         round(recall_score(y_test, preds), 4),
            "train_time_sec": train_time,
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="baseline-model",
            registered_model_name="fake-review-baseline"
        )

        print("\n" + "="*45)
        print("  BASELINE MODEL RESULTS")
        print("="*45)
        for k, v in metrics.items():
            print(f"  {k:<22}: {v}")
        print("="*45)
        print(classification_report(y_test, preds,
              target_names=["Negative", "Positive"]))

    return metrics

if __name__ == "__main__":
    train_baseline()
