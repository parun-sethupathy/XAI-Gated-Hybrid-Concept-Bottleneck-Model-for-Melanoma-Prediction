# base_dataset.py
import torch
from torch.utils.data import Dataset

class BaseConceptDataset(Dataset):
    """
    Abstract Base Class for all CBM Datasets.
    Enforces the contract that all datasets must return the same 4 outputs.
    """
    
    def __init__(self, metadata_df, transforms=None):
        self.df = metadata_df
        self.transforms = transforms
        self.samples = [] 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. image tensor
        # 2. concepts (list of ints)
        # 3. masks (list of floats)
        # 4. label (int: 1 or 0)
        raise NotImplementedError("Child dataset class must implement __getitem__")