# pipelines/deployment_pipeline/optimize.py
import os
import time
import json
import torch
import mlflow
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config.config_loader import mlflow_config, distilbert_config


class ModelBenchmarker:
    """Handles latency benchmarking for any inference function."""

    def __init__(self, warmup_runs: int = 5, benchmark_runs: int = 100):
        self.warmup_runs    = warmup_runs
        self.benchmark_runs = benchmark_runs

    def benchmark(self, fn) -> float:
        for _ in range(self.warmup_runs):
            fn()
        start = time.perf_counter()
        for _ in range(self.benchmark_runs):
            fn()
        return round((time.perf_counter() - start) / self.benchmark_runs * 1000, 3)


class ONNXExporter:
    """Handles ONNX export from a PyTorch model."""

    def __init__(self, opset_version: int = 14):
        self.opset_version = opset_version

    def export(self, model, inputs: dict, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            output_path,
            input_names        = ["input_ids", "attention_mask"],
            output_names       = ["logits"],
            dynamic_axes       = {
                "input_ids":      {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits":         {0: "batch_size"}
            },
            opset_version      = self.opset_version,
            do_constant_folding = True
        )
        print(f"  ONNX model saved to {output_path}")
        return output_path


class INT8Quantizer:
    """Handles dynamic INT8 quantization of a PyTorch model."""

    def quantize(self, model) -> torch.nn.Module:
        return torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

    def save(self, model, tokenizer, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, "model_int8.pt"))
        tokenizer.save_pretrained(save_path)
        print(f"  INT8 model saved to {save_path}")


class ONNXInferenceSession:
    """Wraps an ONNX Runtime session for benchmarking."""

    def __init__(self, onnx_path: str):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, sess_options=sess_options)

    def run(self, ort_inputs: dict):
        return self.session.run(None, ort_inputs)


class ModelOptimizer:
    """
    Orchestrates FP32 benchmarking, INT8 quantization,
    ONNX export, and MLflow logging.
    """

    def __init__(self, model_path: str = "model/distilbert-final"):
        self.model_path  = model_path
        self.ml_cfg      = mlflow_config()
        self.db_cfg      = distilbert_config()
        self.benchmarker = ModelBenchmarker()
        self.exporter    = ONNXExporter(opset_version=14)
        self.quantizer   = INT8Quantizer()

    def _load_model(self):
        print(f"Loading model from: {self.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model     = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        model.eval()
        return tokenizer, model

    def _get_sample_inputs(self, tokenizer) -> dict:
        sample_text = ["This product is absolutely amazing and I love it so much!"]
        return tokenizer(
            sample_text,
            return_tensors = "pt",
            truncation     = True,
            padding        = True,
            max_length     = self.db_cfg["max_length"]
        )

    def _benchmark_fp32(self, model, inputs: dict) -> float:
        print("\n[1/3] Benchmarking FP32 latency...")
        def fp32_inference():
            with torch.no_grad():
                model(**inputs)
        latency = self.benchmarker.benchmark(fp32_inference)
        print(f"  FP32 latency: {latency} ms")
        return latency

    def _benchmark_int8(self, model, inputs: dict, tokenizer) -> float:
        print("\n[2/3] Applying INT8 quantization + benchmarking...")
        model_int8 = self.quantizer.quantize(model)
        model_int8.eval()
        def int8_inference():
            with torch.no_grad():
                model_int8(**inputs)
        latency = self.benchmarker.benchmark(int8_inference)
        print(f"  INT8 latency: {latency} ms")
        self.quantizer.save(model_int8, tokenizer, "model/distilbert-int8")
        return latency

    def _benchmark_onnx(self, model, inputs: dict) -> float:
        print("\n[3/3] Exporting to ONNX + benchmarking...")
        onnx_path = self.exporter.export(model, inputs, "model/model.onnx")
        ort_session = ONNXInferenceSession(onnx_path)
        ort_inputs  = {
            "input_ids":      inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy()
        }
        def onnx_inference():
            ort_session.run(ort_inputs)
        latency = self.benchmarker.benchmark(onnx_inference)
        print(f"  ONNX latency: {latency} ms")
        return latency

    def _compute_metrics(self, fp32: float, int8: float, onnx: float) -> dict:
        return {
            "fp32_latency_ms":              fp32,
            "int8_latency_ms":              int8,
            "onnx_latency_ms":              onnx,
            "latency_reduction_int8_pct":   round((fp32 - int8) / fp32 * 100, 2),
            "latency_reduction_onnx_pct":   round((fp32 - onnx) / fp32 * 100, 2),
            "int8_speedup":                 round(fp32 / int8, 2),
            "onnx_speedup":                 round(fp32 / onnx, 2),
        }

    def _log_to_mlflow(self, metrics: dict):
        mlflow.set_tracking_uri(self.ml_cfg["tracking_uri"])
        mlflow.set_experiment(self.ml_cfg["experiment_name"])
        run_name = f"optimization-{time.strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "base_model_path": self.model_path,
                "quantization":    "dynamic_int8",
                "onnx_opset":      14,
                "benchmark_runs":  self.benchmarker.benchmark_runs,
                "max_length":      self.db_cfg["max_length"]
            })
            mlflow.log_metrics(metrics)
            mlflow.log_artifact("model/model.onnx", artifact_path="onnx-model")

    def _save_metrics(self, metrics: dict):
        os.makedirs("model", exist_ok=True)
        with open("model/optimization_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("  Saved to model/optimization_metrics.json")

    def _print_summary(self, metrics: dict):
        print("\n" + "="*50)
        print("  OPTIMIZATION RESULTS")
        print("="*50)
        print(f"  FP32 latency : {metrics['fp32_latency_ms']} ms  (baseline)")
        print(f"  INT8 latency : {metrics['int8_latency_ms']} ms  "
              f"({metrics['latency_reduction_int8_pct']}% faster)")
        print(f"  ONNX latency : {metrics['onnx_latency_ms']} ms  "
              f"({metrics['latency_reduction_onnx_pct']}% faster)")
        print("="*50)

    def run(self) -> dict:
        print("\n--- MODEL OPTIMIZATION ---")
        tokenizer, model = self._load_model()
        inputs           = self._get_sample_inputs(tokenizer)

        fp32_latency = self._benchmark_fp32(model, inputs)
        int8_latency = self._benchmark_int8(model, inputs, tokenizer)
        onnx_latency = self._benchmark_onnx(model, inputs)

        metrics = self._compute_metrics(fp32_latency, int8_latency, onnx_latency)

        self._log_to_mlflow(metrics)
        self._save_metrics(metrics)
        self._print_summary(metrics)

        return metrics


if __name__ == "__main__":
    optimizer = ModelOptimizer()
    optimizer.run()
