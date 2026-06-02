# data_loaders.py
import os
import pandas as pd
import numpy as np
import torch
from PIL import Image

from base_dataset import BaseConceptDataset

class Derm7ptDataset(BaseConceptDataset):
    def __init__(self, metadata_df, image_dir, transforms=None):
        super().__init__(metadata_df, transforms)
        self.image_dir = image_dir
        self.concept_cols = [
            "pigment_network", "streaks", "pigmentation", 
            "regression_structures", "dots_and_globules", 
            "blue_whitish_veil", "vascular_structures"
        ]
        self._prepare_samples()

    def _parse_concept(self, val, col_idx):
        """ Translates medical text ('absent', 'typical') into integers (0, 1, 2) """
        # 1. Handle Missing Data
        if pd.isna(val) or val == -1 or str(val).strip() == '-1' or str(val).lower() == 'unknown': 
            return 0, 0.0  # Returns (concept_value, mask_value)
            
        # 2. If it is already a number
        if isinstance(val, (int, float)):
            return int(val), 1.0
            
        # 3. If it is a real string from the CSV
        val_str = str(val).lower().strip()
        
        # Class 0 (Negative)
        if val_str in ['absent', 'none', 'false', '0']:
            return 0, 1.0
            
        # Class 1 (Positive / Regular)
        if val_str in ['present', 'typical', 'regular', 'true', '1']:
            return 1, 1.0
            
        # Class 2 (Atypical / Irregular)
        if val_str in ['atypical', 'irregular', '2']:
            # In our config, only blue_whitish_veil (Index 5) has 3 dimensions. 
            # If we get 'irregular' on a 2-dim head, we force it to 1 to prevent crashes.
            if col_idx == 5: 
                return 2, 1.0
            else:
                return 1, 1.0 
                
        # Failsafe fallback
        return 1, 1.0

    def _prepare_samples(self):
        for _, row in self.df.iterrows():
            img_path = os.path.join(self.image_dir, str(row.get('image_path', '')))
            
            concepts, masks = [], []
            for i, col in enumerate(self.concept_cols):
                val = row.get(col, np.nan)
                c_val, c_mask = self._parse_concept(val, i)
                concepts.append(c_val)
                masks.append(c_mask)
                
            label = 1 if str(row.get('diagnosis')).lower() == 'melanoma' else 0
            self.samples.append((img_path, concepts, masks, label))

    def __getitem__(self, idx):
        img_path, concepts, masks, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            img = Image.new('RGB', (224, 224)) 
        if self.transforms:
            img = self.transforms(img)
        return (
            img, 
            torch.tensor(concepts, dtype=torch.long), 
            torch.tensor(masks, dtype=torch.float32), 
            torch.tensor(label, dtype=torch.float32)
        )


class HAM10000Dataset(BaseConceptDataset):
    """ Zero-Shot Generalization Loader """
    def __init__(self, metadata_df, image_paths_dict, transforms=None):
        super().__init__(metadata_df, transforms)
        self.image_paths_dict = image_paths_dict
        self._prepare_samples()

    def _prepare_samples(self):
        for _, row in self.df.iterrows():
            img_id = row['image_id']
            img_path = self.image_paths_dict.get(img_id)
            if not img_path: continue 
            
            concepts = [0] * 7 
            masks = [0.0] * 7  
            label = 1 if row.get('dx') == 'mel' else 0
            self.samples.append((img_path, concepts, masks, label))

    def __getitem__(self, idx):
        img_path, concepts, masks, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            img = Image.new('RGB', (224, 224))
        if self.transforms:
            img = self.transforms(img)
        return (
            img, 
            torch.tensor(concepts, dtype=torch.long), 
            torch.tensor(masks, dtype=torch.float32), 
            torch.tensor(label, dtype=torch.float32)
        )