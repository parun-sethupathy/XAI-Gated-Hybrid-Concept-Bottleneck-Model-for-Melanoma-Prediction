# config.py
import torch

class Config:
    # 1. Global Settings
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Model Hyperparameters
    BACKBONE = "tf_efficientnetv2_s"
    FEAT_PROJ_DIM = 256
    HIDDEN_DIM = 256
    DROPOUT = 0.30

    # 3. Training Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Loss Balancing (Phase 2)
    LAMBDA_CONCEPT = 1.0
    LAMBDA_TASK = 1.0
    LAMBDA_IMG = 0.5
    LAMBDA_CBM = 1.0
    
    # Focal Loss Settings (Prioritizing Recall)
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA_MELANOMA = 0.85 
    
    # 4. Dataset Geometries
    DERM7PT_CONCEPT_DIMS = [2, 2, 2, 2, 2, 3, 2] 
    HAM10000_CONCEPT_DIMS = [2, 2, 2, 2]
    
    # Active Dataset Toggle 
    ACTIVE_DATASET = "derm7pt" 

    @classmethod
    def get_active_concept_dims(cls):
        if cls.ACTIVE_DATASET == "derm7pt":
            return cls.DERM7PT_CONCEPT_DIMS
        elif cls.ACTIVE_DATASET == "ham10000":
            return cls.HAM10000_CONCEPT_DIMS
        else:
            raise ValueError("Unknown dataset configuration.")