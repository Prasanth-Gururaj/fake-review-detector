# pipelines/data_pipeline/validate.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
from config.config_loader import data_config

def validate(df: pd.DataFrame) -> bool:
    cfg = data_config()
    checks = {
        "no_nulls":        df["text"].isnull().sum() == 0,
        "has_both_labels": df["label"].nunique() == 2,
        "min_length":      (df["text"].str.len() > 10).all(),
        "min_rows":        len(df) >= 1000,
    }
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")
    all_passed = all(checks.values())
    if not all_passed:
        raise ValueError("Data validation failed. Fix issues before training.")
    print("All data validation checks passed.")
    return True
