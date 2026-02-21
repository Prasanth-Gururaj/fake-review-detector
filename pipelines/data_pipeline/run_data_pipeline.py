import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from pipelines.data_pipeline.ingest import ingest_amazon_polarity
from pipelines.data_pipeline.preprocess import preprocess, split_data
from pipelines.data_pipeline.validate import validate

def run():
    print('--- DATA PIPELINE STARTED ---')
    df = ingest_amazon_polarity(n_samples=50000)
    df = preprocess(df)
    validate(df)
    train_df, test_df = split_data(df)
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    df.to_csv('data/reviews.csv', index=False)
    print('--- DATA PIPELINE COMPLETE ---')
    print(f'Files saved: data/train.csv, data/test.csv, data/reviews.csv')

if __name__ == '__main__':
    run()