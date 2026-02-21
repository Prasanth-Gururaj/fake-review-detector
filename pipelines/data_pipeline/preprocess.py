import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', '', text)       # remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip() # normalize whitespace
    return text

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print('Cleaning text...')
    df['text'] = df['text'].apply(clean_text)
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.len() > 10]
    return df

def split_data(df: pd.DataFrame):
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f'Train: {len(train)} | Test: {len(test)}')
    return train, test