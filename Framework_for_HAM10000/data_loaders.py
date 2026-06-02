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
        if pd.isna(val) or val == -1 or str(val).strip() == '-1' or str(val).lower() == 'unknown': 
            return 0, 0.0  
        if isinstance(val, (int, float)): return int(val), 1.0
        val_str = str(val).lower().strip()
        if val_str in ['absent', 'none', 'false', '0']: return 0, 1.0
        if val_str in ['present', 'typical', 'regular', 'true', '1']: return 1, 1.0
        if val_str in ['atypical', 'irregular', '2']: return 2, 1.0 if col_idx == 5 else (1, 1.0)
        return 1, 1.0

    def _prepare_samples(self):
        for _, row in self.df.iterrows():
            concepts, masks = [], []
            for i, col in enumerate(self.concept_cols):
                val = row.get(col, np.nan)
                c_val, c_mask = self._parse_concept(val, i)
                concepts.append(c_val)
                masks.append(c_mask)
                
            label = 1 if str(row.get('diagnosis')).lower().strip() == 'melanoma' else 0
            
            derm_img, clinic_img = row.get('derm'), row.get('clinic')
            if pd.notna(derm_img) and str(derm_img).strip() != '':
                self.samples.append((os.path.join(self.image_dir, str(derm_img)), concepts, masks, label))
            if pd.notna(clinic_img) and str(clinic_img).strip() != '':
                self.samples.append((os.path.join(self.image_dir, str(clinic_img)), concepts, masks, label))

    def __getitem__(self, idx):
        img_path, concepts, masks, label = self.samples[idx]
        try:
            if os.path.isdir(img_path): raise IsADirectoryError()
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, IsADirectoryError):
            img = Image.new('RGB', (224, 224)) 
        if self.transforms: img = self.transforms(img)
        return (
            img, torch.tensor(concepts, dtype=torch.long), 
            torch.tensor(masks, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        )

class HAM10000Dataset(BaseConceptDataset):
    def __init__(self, metadata_df, image_dir, transforms=None):
        super().__init__(metadata_df, transforms)
        self.image_dir = image_dir
        
        # Dynamically index every .jpg in the HAM10000 folder regardless of sub-directory structure
        self.image_paths = {}
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.jpg'):
                    img_id = file.replace('.jpg', '')
                    self.image_paths[img_id] = os.path.join(root, file)
        
        self._prepare_samples()

    def _prepare_samples(self):
        for _, row in self.df.iterrows():
            img_id = str(row['image_id']).strip()
            if img_id not in self.image_paths: continue 
            
            img_path = self.image_paths[img_id]
            concepts = [0] * 7 
            masks = [0.0] * 7  
            label = 1 if str(row.get('dx')).lower().strip() == 'mel' else 0
            
            self.samples.append((img_path, concepts, masks, label))

    def __getitem__(self, idx):
        img_path, concepts, masks, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            img = Image.new('RGB', (224, 224))
        if self.transforms: img = self.transforms(img)
        return (
            img, torch.tensor(concepts, dtype=torch.long), 
            torch.tensor(masks, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        )
