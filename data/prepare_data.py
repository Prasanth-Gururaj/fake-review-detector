from datasets import load_dataset
import pandas as pd
import os

os.makedirs('data', exist_ok=True)

print('Loading Amazon Polarity dataset (50,000 samples)...')
amazon = load_dataset('amazon_polarity', split='train').shuffle(seed=42).select(range(50000))

df = pd.DataFrame({
    'text':    amazon['content'],
    'label':   amazon['label'],
    'is_fake': 0
})

df.to_csv('data/reviews.csv', index=False)
print(f'Saved {len(df)} reviews to data/reviews.csv')
print(df.head())
print(df['label'].value_counts())