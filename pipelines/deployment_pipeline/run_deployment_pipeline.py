# pipelines/deployment_pipeline/run_deployment_pipeline.py
import json
import os
from pipelines.deployment_pipeline.optimize import ModelOptimizer


class DeploymentPipeline:
    """Orchestrates all deployment steps: optimize → (future: serve, monitor)."""

    def __init__(self, model_path: str = "model/distilbert-final"):
        self.model_path = model_path
        self.optimizer  = ModelOptimizer(model_path=model_path)

    def run(self) -> dict:
        print("=" * 60)
        print("  DEPLOYMENT PIPELINE STARTED")
        print("=" * 60)

        print("\n[1/1] Optimizing model (INT8 + ONNX)...")
        opt_metrics = self.optimizer.run()

        os.makedirs("model", exist_ok=True)
        with open("model/optimization_metrics.json", "w") as f:
            json.dump(opt_metrics, f, indent=2)
        print("Saved to model/optimization_metrics.json")

        self._print_summary(opt_metrics)
        return opt_metrics

    def _print_summary(self, metrics: dict):
        print("\n" + "=" * 60)
        print("  DEPLOYMENT PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  FP32  latency : {metrics['fp32_latency_ms']} ms")
        print(f"  INT8  latency : {metrics['int8_latency_ms']} ms  "
              f"({metrics['latency_reduction_int8_pct']}% faster)")
        print(f"  ONNX  latency : {metrics['onnx_latency_ms']} ms  "
              f"({metrics['latency_reduction_onnx_pct']}% faster)")
        print("=" * 60)


if __name__ == "__main__":
    pipeline = DeploymentPipeline(model_path="model/distilbert-final")
    pipeline.run()
