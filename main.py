# main.py
import argparse
from pipelines.training_pipeline.run_training_pipeline   import TrainingPipeline
from pipelines.deployment_pipeline.run_deployment_pipeline import DeploymentPipeline


def main():
    parser = argparse.ArgumentParser(description="Fake Review Detector — Pipeline Runner")
    parser.add_argument(
        "--pipeline",
        choices=["train", "deploy", "all"],
        default="train",
        help="Which pipeline to run: train | deploy | all"
    )
    parser.add_argument(
        "--model-path",
        default="model/distilbert-final",
        help="Path to saved model for deployment pipeline"
    )
    args = parser.parse_args()

    if args.pipeline in ("train", "all"):
        print("\n🚀 Running Training Pipeline...")
        training_pipeline = TrainingPipeline()
        training_pipeline.run()

    if args.pipeline in ("deploy", "all"):
        print("\n🚀 Running Deployment Pipeline...")
        deployment_pipeline = DeploymentPipeline(model_path=args.model_path)
        deployment_pipeline.run()


if __name__ == "__main__":
    main()
