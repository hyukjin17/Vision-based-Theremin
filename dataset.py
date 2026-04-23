"""
Hyuk Jin Chung
4/22/2026

Data loader for MLP training
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

class GestureDataset(Dataset):
    """Dataloader class"""
    def __init__(self, csv_file, split_type='train'):
        """
        Args:
            csv_file: Filepath to the split CSV file
            split_type: 'train' or 'test'
        """
        df = pd.read_csv(csv_file)
        
        # Filter dataframe by train or test
        df = df[df['split'] == split_type]
        
        # Extract landmark features (ignoring 'label' and 'split' columns)
        feature_cols = [col for col in df.columns if col not in ['label', 'split']]
        
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df['label'].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]