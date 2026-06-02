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
    pl.seed_everything(42)
    print("🚀 Initializing Phase 1.5: The Concept Forge...")

    DATA_DIR = "/content/drive/MyDrive/XAI_Cloud_Run/derm7pt"  
    META_CSV = os.path.join(DATA_DIR, "meta", "meta.csv") 
    IMAGE_DIR = os.path.join(DATA_DIR, "images")  
    CHECKPOINT_DIR = "/content/drive/MyDrive/XAI_Cloud_Run/cloud_checkpoints/"

    full_df = pd.read_csv(META_CSV)
    
    clean_diag = full_df['diagnosis'].astype(str).str.lower().str.strip()
    binary_labels = clean_diag.str.contains('melanoma').astype(int)
    
    train_df, val_df = train_test_split(
        full_df, test_size=0.2, stratify=binary_labels, random_state=42
    )

    print("⚖️ Calculating Class Weights...")
    train_clean_diag = train_df['diagnosis'].astype(str).str.lower().str.strip()
    is_melanoma = train_clean_diag.str.contains('melanoma')
    count_melanoma = is_melanoma.sum()
    count_other = len(train_df) - count_melanoma
    
    weight_melanoma = 1.0 / count_melanoma if count_melanoma > 0 else 1.0
    weight_other = 1.0 / count_other if count_other > 0 else 1.0
    sample_weights = [weight_melanoma if val else weight_other for val in is_melanoma]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights), 
        num_samples=len(sample_weights), 
        replacement=True
    )

    # ---------------------------------------------------------
    # AGGRESSIVE AUGMENTATION
    # ---------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])

    # Validation must remain pristine
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Instantiate datasets with new dual-image loading and transforms
    train_dataset = Derm7ptDataset(metadata_df=train_df, image_dir=IMAGE_DIR, transforms=train_transform)
    val_dataset = Derm7ptDataset(metadata_df=val_df, image_dir=IMAGE_DIR, transforms=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, num_workers=2) 
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    print("🧠 Building GatedHybridCBM Architecture...")
    model = GatedHybridCBM(
        concept_dims=Config.get_active_concept_dims(),
        backbone=Config.BACKBONE,
        feat_proj_dim=Config.FEAT_PROJ_DIM,
        hidden=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT
    )
    lightning_system = CBM_System(model, Config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="forged-cbm-{epoch:02d}-{val_auc:.3f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_auc",
        min_delta=0.005,
        patience=10, # Increased patience because augmentation slows down convergence
        verbose=True,
        mode="max"
    )

    logger = TensorBoardLogger("/content/drive/MyDrive/XAI_Cloud_Run/tb_logs", name="GatedHybridCBM_Forged")

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",     
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision="16-mixed",   
        log_every_n_steps=10
    )

    print("🔥 Launching The Concept Forge...")
    trainer.fit(lightning_system, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"✅ Training Complete. Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
