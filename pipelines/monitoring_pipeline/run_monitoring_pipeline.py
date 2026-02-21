import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pipelines.monitoring_pipeline.drift_monitor import run_drift_check
from pipelines.monitoring_pipeline.performance_monitor import get_performance_summary

def run():
    print('--- MONITORING PIPELINE STARTED ---')
    summary = get_performance_summary()
    print(f'Performance Summary: {summary}')
    drift_detected = run_drift_check()
    print(f'Drift Detected: {drift_detected}')
    if drift_detected:
        print('ACTION: Trigger retraining pipeline')
    print('--- MONITORING PIPELINE COMPLETE ---')

if __name__ == '__main__':
    run()