import numpy as np
import torch
from torch.utils.data import Dataset

class SklearnPipelineDataset(Dataset):
    def __init__(self, X: np.ndarray|torch.Tensor, y: np.ndarray|torch.Tensor, sklearn_pipeline, fit=False):
        # Apply the sklearn pipeline transformations
        if fit:
            self.X = torch.tensor(sklearn_pipeline.fit_transform(X), dtype=torch.float32)
        else:
            self.X = torch.tensor(sklearn_pipeline.transform(X), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]