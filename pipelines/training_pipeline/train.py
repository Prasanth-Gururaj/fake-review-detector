# pipelines/training_pipeline/train.py
from datetime import datetime
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import torch
import time
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback)
from datasets import Dataset
from sklearn.metrics import (f1_score, accuracy_score,
                             precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from config.config_loader import (data_config, mlflow_config,
                                   baseline_config, distilbert_config)



# ✅ PyfFunc wrapper — avoids AMP pickle error, populates Models column
class DistilBertWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["model_path"])
        self.model     = AutoModelForSequenceClassification.from_pretrained(context.artifacts["model_path"])
        self.model.eval()

    def predict(self, context, model_input):
        import torch
        import numpy as np
        texts  = model_input["text"].tolist()
        inputs = self.tokenizer(texts, return_tensors="pt",
                                truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return np.argmax(logits.numpy(), axis=1)



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy":  round(float(accuracy_score(labels, preds)), 4),
        "f1":        round(float(f1_score(labels, preds)), 4),
        "precision": round(float(precision_score(labels, preds)), 4),
        "recall":    round(float(recall_score(labels, preds)), 4),
    }



def train_baseline():
    d_cfg  = data_config()
    ml_cfg = mlflow_config()
    b_cfg  = baseline_config()

    mlflow.set_tracking_uri(ml_cfg["tracking_uri"])
    mlflow.set_experiment(ml_cfg["experiment_name"])

    print("\n--- BASELINE MODEL ---")
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
            pipeline, "baseline-model",
            registered_model_name="fake-review-baseline"
        )
        print(f"Baseline F1:       {metrics['f1_score']}")
        print(f"Baseline Accuracy: {metrics['accuracy']}")
        print(f"Train Time:        {train_time}s")
    return metrics



def train_distilbert():
    d_cfg  = data_config()
    ml_cfg = mlflow_config()
    db_cfg = distilbert_config()

    mlflow.set_tracking_uri(ml_cfg["tracking_uri"])
    mlflow.set_experiment(ml_cfg["experiment_name"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- DISTILBERT FINE-TUNING ---")
    print(f"Device : {device}")
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {round(torch.cuda.get_device_properties(0).total_memory/1e9, 1)} GB")

    train_df = pd.read_csv(d_cfg["train_path"])
    test_df  = pd.read_csv(d_cfg["test_path"])

    TRAIN_SIZE = 10000
    TEST_SIZE  = 2000

    train_data = (train_df[["text", "label"]]
                  .rename(columns={"label": "labels"})
                  .sample(TRAIN_SIZE, random_state=42)
                  .reset_index(drop=True))
    test_data  = (test_df[["text", "label"]]
                  .rename(columns={"label": "labels"})
                  .sample(TEST_SIZE, random_state=42)
                  .reset_index(drop=True))

    print(f"Train samples: {TRAIN_SIZE} | Test samples: {TEST_SIZE}")
    print("Tokenizing...")

    tokenizer = AutoTokenizer.from_pretrained(db_cfg["model_name"])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=db_cfg["max_length"]
        )

    train_ds = Dataset.from_pandas(train_data).map(tokenize, batched=True)
    test_ds  = Dataset.from_pandas(test_data).map(tokenize, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        db_cfg["model_name"],
        num_labels=db_cfg["num_labels"]
    )

    training_args = TrainingArguments(
        output_dir                  = "model/distilbert-checkpoints",
        num_train_epochs            = db_cfg["epochs"],
        per_device_train_batch_size = db_cfg["batch_size"],
        per_device_eval_batch_size  = db_cfg["batch_size"],
        learning_rate               = db_cfg["learning_rate"],
        warmup_steps                = db_cfg["warmup_steps"],
        weight_decay                = db_cfg["weight_decay"],
        eval_strategy               = db_cfg["eval_strategy"],
        save_strategy               = db_cfg["save_strategy"],
        load_best_model_at_end      = db_cfg["load_best_model_at_end"],
        metric_for_best_model       = "f1",
        logging_steps               = 50,
        report_to                   = "none",
        fp16                        = True,
        dataloader_num_workers      = 0
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = test_ds,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Unique run name per execution
    run_name = f"distilbert-finetuned-v{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as active_run:
        run_id = active_run.info.run_id

        mlflow.log_params(db_cfg)
        mlflow.log_params({
            "train_samples": TRAIN_SIZE,
            "test_samples":  TEST_SIZE,
            "device":        device,
            "fp16":          True
        })

        print("Training — ~15 mins on RTX 4060...")
        print("Watch: http://127.0.0.1:5000\n")
        start = time.time()
        trainer.train()
        train_time = round(time.time() - start, 2)

        print("\nEvaluating...")
        eval_results = trainer.evaluate()

        final_metrics = {
            "accuracy":       round(eval_results.get("eval_accuracy", 0), 4),
            "f1_score":       round(eval_results.get("eval_f1", 0), 4),
            "precision":      round(eval_results.get("eval_precision", 0), 4),
            "recall":         round(eval_results.get("eval_recall", 0), 4),
            "eval_loss":      round(eval_results.get("eval_loss", 0), 4),
            "train_time_sec": train_time,
            "run_id":         run_id,
            "run_name":       run_name,
        }
        mlflow.log_metrics({k: v for k, v in final_metrics.items()
                            if k not in ("run_id", "run_name")})

        # Save HuggingFace model + tokenizer to disk
        os.makedirs("model/distilbert-final", exist_ok=True)
        trainer.model.save_pretrained("model/distilbert-final")
        tokenizer.save_pretrained("model/distilbert-final")
        print("Model saved to model/distilbert-final/")

        # ✅ Log as pyfunc — populates Models column + registers in Model Registry
        mlflow.pyfunc.log_model(
            artifact_path    = "distilbert-model",
            python_model     = DistilBertWrapper(),
            artifacts        = {"model_path": os.path.abspath("model/distilbert-final")},
            registered_model_name = "fake-review-distilbert"
        )

        print("\n" + "="*50)
        print("  DISTILBERT RESULTS")
        print("="*50)
        for k, v in final_metrics.items():
            print(f"  {k:<22}: {v}")
        print("="*50)

    return final_metrics  # includes run_id and run_name for register_model.py



if __name__ == "__main__":
    train_distilbert()
    # train_baseline()
