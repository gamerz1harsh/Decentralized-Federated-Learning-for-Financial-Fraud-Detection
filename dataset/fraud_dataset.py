import pandas as pd
import torch
from torch.utils.data import Dataset


class FraudDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        if "Class" not in self.df.columns:
            raise ValueError(f"Expected a 'Class' column in {csv_path}")
        self.features = self.df.drop(columns=["Class"])
        self.labels = self.df["Class"]
        if self.features.empty or not self.features.select_dtypes(include="number").columns.equals(self.features.columns):
            raise ValueError("All feature columns must be numeric")
        if self.df.isna().any().any():
            raise ValueError(f"Missing values found in {csv_path}")
        self.x = torch.tensor(self.features.values, dtype=torch.float32)
        self.y = torch.tensor(self.labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

