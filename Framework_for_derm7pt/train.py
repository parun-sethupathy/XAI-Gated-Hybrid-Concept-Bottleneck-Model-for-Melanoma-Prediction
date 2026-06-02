# train.py
import os
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import Config
from data_loaders import Derm7ptDataset
from models import GatedHybridCBM
from engine import CBM_System

def main():
    # 1. Cloud Reproducibility Lock
    pl.seed_everything(42)
    print("Initializing Cloud Training Sequence...")

    # 2. Data Loading & Splits
    # IMPORTANT: When you move to AWS, you will change this DATA_DIR path to your cloud storage path!
    DATA_DIR = r"C:\Users\prune\OneDrive\Desktop\XAI CBM Coalations\derm7pt"  
    META_CSV = os.path.join(DATA_DIR, "meta", "meta.csv") 
    IMAGE_DIR = os.path.join(DATA_DIR, "images")  

    full_df = pd.read_csv(META_CSV)
    
    # Strict Stratified Split (80% Train, 20% Val)
    train_df, val_df = train_test_split(
        full_df, test_size=0.2, stratify=full_df['diagnosis'], random_state=42
    )

    # ---------------------------------------------------------
    # 3. MITIGATION 3: The Weighted Random Sampler 
    # (Forces 50/50 batches to defeat mode collapse)
    # ---------------------------------------------------------
    print("Calculating Class Weights...")
    is_melanoma = train_df['diagnosis'].str.lower() == 'melanoma'
    count_melanoma = is_melanoma.sum()
    count_other = len(train_df) - count_melanoma
    
    weight_melanoma = 1.0 / count_melanoma if count_melanoma > 0 else 1.0
    weight_other = 1.0 / count_other if count_other > 0 else 1.0

    sample_weights = [weight_melanoma if label == 'melanoma' else weight_other 
                      for label in train_df['diagnosis'].str.lower()]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights), 
        num_samples=len(sample_weights), 
        replacement=True
    )

    # 4. DataLoaders
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = Derm7ptDataset(metadata_df=train_df, image_dir=IMAGE_DIR, transforms=basic_transform)
    val_dataset = Derm7ptDataset(metadata_df=val_df, image_dir=IMAGE_DIR, transforms=basic_transform)

    # Note: Shuffle MUST be False when using a custom sampler!
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, num_workers=4) 
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # 5. Architecture & Engine
    print("\nBuilding GatedHybridCBM Architecture...")
    model = GatedHybridCBM(
        concept_dims=Config.get_active_concept_dims(),
        backbone=Config.BACKBONE,
        feat_proj_dim=Config.FEAT_PROJ_DIM,
        hidden=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT
    )
    lightning_system = CBM_System(model, Config)

    # ---------------------------------------------------------
    # 6. CLOUD FAILSAFES (Protecting your AWS Budget)
    # ---------------------------------------------------------
    # Failsafe 1: Save the absolute best model based on AUC, not just the last epoch.
    checkpoint_callback = ModelCheckpoint(
        dirpath="cloud_checkpoints/",
        filename="gated-cbm-{epoch:02d}-{val_auc:.3f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
    )

    # Failsafe 2: Kill the run if AUC flatlines for 7 straight epochs
    early_stop_callback = EarlyStopping(
        monitor="val_auc",
        min_delta=0.005,
        patience=7,
        verbose=True,
        mode="max"
    )

    # 7. Cloud Logger
    logger = TensorBoardLogger("tb_logs", name="GatedHybridCBM_Derm7pt")

    # 8. The Cloud Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",     # Automatically finds your Cloud GPU
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision="16-mixed",   # Half-precision memory saving (Speeds up AWS GPUs by 2x)
        log_every_n_steps=10
    )

    # 9. IGNITION
    print("\nLaunching Training Sequence...")
    trainer.fit(lightning_system, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Training Complete. Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()