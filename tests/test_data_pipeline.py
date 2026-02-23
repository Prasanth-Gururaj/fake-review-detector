import pandas as pd
from pipelines.data_pipeline.preprocess import clean_text, preprocess, split_data
from pipelines.data_pipeline.validate import validate

def test_clean_text_removes_html():
    assert clean_text('<b>Hello</b>') == 'Hello'

def test_clean_text_normalizes_whitespace():
    assert clean_text('Hello   world') == 'Hello world'

def test_validate_passes_good_data():
    df = pd.DataFrame({
        'text':  ['This product works great'] * 1000 + ['Terrible quality item'] * 1000,
        'label': [1] * 1000 + [0] * 1000
    })
    assert validate(df) == True


def test_split_ratio():
    df = pd.DataFrame({'text': ['review'] * 1000, 'label': [1]*500 + [0]*500})
    train, test = split_data(df)
    assert len(test) == 200
    assert len(train) == 800