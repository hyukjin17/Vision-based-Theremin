"""
Hyuk Jin Chung
4/22/2026

Splits the collected data into train and test data (90/10 split)
Data is split up by class to make sure that each class gets a separate test set
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_dataset(input_csv, output_csv):
    """Load the dataset csv and split the data into test/train sets"""
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Data is split by data label to get exactly 90/10 split per class
    train_df, test_df = train_test_split(
        df, 
        test_size=0.10, 
        stratify=df['label'], 
        random_state=42 # fixed seed for reproducibility
    )
    
    # Add the split identifier
    train_df = train_df.copy()
    train_df['split'] = 'train'
    test_df = test_df.copy()
    test_df['split'] = 'test'
    
    # Recombine and save
    final_df = pd.concat([train_df, test_df]).sort_index()
    final_df.to_csv(output_csv, index=False)
    
    print("\nDataset Split Summary:")
    print(final_df.groupby(['label', 'split']).size())
    print(f"\nSaved processed dataset to {output_csv}")

if __name__ == "__main__":
    prepare_dataset('gesture_dataset.csv', 'gesture_dataset_split.csv')