# pipelines/training_pipeline/register_model.py
import mlflow
from mlflow.tracking import MlflowClient
from config.config_loader import mlflow_config


def register_best_model(train_metrics: dict) -> dict:
    ml_cfg = mlflow_config()
    mlflow.set_tracking_uri(ml_cfg["tracking_uri"])
    client = MlflowClient()

    experiment = client.get_experiment_by_name(ml_cfg["experiment_name"])
    if experiment is None:
        print("No experiment found.")
        return {}

    # Fetch all distilbert runs sorted by F1
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName LIKE 'distilbert-finetuned%'",
        order_by=["metrics.f1_score DESC"],
    )

    if not runs:
        print("No distilbert runs found to register.")
        return {}

    # Fetch baseline for comparison
    baseline_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName LIKE 'baseline%'",
        order_by=["metrics.f1_score DESC"],
    )
    baseline_f1 = baseline_runs[0].data.metrics.get("f1_score", 0) if baseline_runs else 0

    print("\n" + "="*55)
    print("  MODEL COMPARISON (sorted by F1)")
    print("="*55)
    for run in runs:
        name   = run.data.tags.get("mlflow.runName", run.info.run_id)
        f1     = run.data.metrics.get("f1_score", 0)
        marker = " <- BEST" if run == runs[0] else ""
        print(f"  {name:<45} F1={f1:.4f}{marker}")
    if baseline_runs:
        name = baseline_runs[0].data.tags.get("mlflow.runName", "baseline")
        print(f"  {name:<45} F1={baseline_f1:.4f}")
    print("="*55)

    best_run    = runs[0]
    best_run_id = best_run.info.run_id
    best_f1     = best_run.data.metrics.get("f1_score", 0)
    best_name   = best_run.data.tags.get("mlflow.runName", best_run_id)

    print(f"\nBest model : {best_name}")
    print(f"Best F1    : {best_f1:.4f}")
    print(f"Improvement over baseline: +{best_f1 - baseline_f1:.4f}")

    # Register best model and promote to Production
    print("Promoting to Production...")
    model_uri  = f"runs:/{best_run_id}/distilbert-model"
    registered = mlflow.register_model(
        model_uri=model_uri,
        name="fake-review-distilbert"
    )

    client.transition_model_version_stage(
        name="fake-review-distilbert",
        version=registered.version,
        stage="Production",
        archive_existing_versions=True    # demotes all previous Production versions
    )

    print(f"Registered: fake-review-distilbert v{registered.version} → Production ✅")

    return {
        "best_run_id":                best_run_id,
        "best_run_name":              best_name,
        "best_f1":                    best_f1,
        "version":                    registered.version,
        "improvement_over_baseline":  round(best_f1 - baseline_f1, 4),
    }
