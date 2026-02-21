# Day 3 - Evaluate model and log metrics to MLflow
import mlflow
from sklearn.metrics import f1_score, accuracy_score, classification_report

def evaluate(model, X_test, y_test, run_name: str = 'evaluation'):
    print('Evaluation pipeline - implement on Day 3')
    raise NotImplementedError('Implement on Day 3')