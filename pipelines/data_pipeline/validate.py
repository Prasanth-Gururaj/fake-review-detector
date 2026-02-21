import pandas as pd

def validate(df: pd.DataFrame) -> bool:
    checks = {
        'no_nulls':        df['text'].isnull().sum() == 0,
        'has_both_labels': df['label'].nunique() == 2,
        'min_length':      (df['text'].str.len() > 10).all(),
        'min_rows':        len(df) >= 1000,
    }
    for check, passed in checks.items():
        status = 'PASS' if passed else 'FAIL'
        print(f'  [{status}] {check}')
    all_passed = all(checks.values())
    if not all_passed:
        raise ValueError('Data validation failed. Fix issues before training.')
    print('All data validation checks passed.')
    return True