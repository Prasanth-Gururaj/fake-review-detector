# tests/test_training_pipeline.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import pytest

def test_train_csv_exists():
    assert os.path.exists("data/train.csv"), "Run data pipeline first"

def test_test_csv_exists():
    assert os.path.exists("data/test.csv"), "Run data pipeline first"

def test_data_has_correct_columns():
    df = pd.read_csv("data/train.csv")
    assert "text"  in df.columns
    assert "label" in df.columns

def test_labels_are_binary():
    df = pd.read_csv("data/train.csv")
    assert set(df["label"].unique()).issubset({0, 1})

def test_no_null_text():
    df = pd.read_csv("data/train.csv")
    assert df["text"].isnull().sum() == 0
