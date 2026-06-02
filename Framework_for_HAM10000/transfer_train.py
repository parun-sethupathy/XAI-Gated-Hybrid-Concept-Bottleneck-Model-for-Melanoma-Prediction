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
from data_loaders import HAM10000Dataset
from models import GatedHybridCBM
from engine import CBM_System

def main():
    pl.seed_everything(42)
    print("Initializing Phase 2: The Brain Transplant (HAM10000)...")

    HAM_DIR = "/content/drive/MyDrive/XAI_Cloud_Run/HAM10000"
    CHECKPOINT_DIR = "/content/drive/MyDrive/XAI_Cloud_Run/cloud_checkpoints/"
    
    meta_files = [f for f in os.listdir(HAM_DIR) if f.endswith('.csv')]
    if not meta_files: raise FileNotFoundError("Could not find metadata CSV in HAM10000 directory.")
    META_CSV = os.path.join(HAM_DIR, meta_files[0])

    print("Hunting for the base Concept Forge Checkpoint...")
    base_model = GatedHybridCBM(
        concept_dims=Config.get_active_concept_dims(),
        backbone=Config.BACKBONE,
        feat_proj_dim=Config.FEAT_PROJ_DIM,
        hidden=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT
    )
    
    if not os.path.exists(CHECKPOINT_DIR):
        raise FileNotFoundError(f"Checkpoint directory missing: {CHECKPOINT_DIR}")
        
    forged_files = [f for f in os.listdir(CHECKPOINT_DIR) if "forged-cbm" in f and f.endswith(".ckpt")]
    if not forged_files:
        raise FileNotFoundError("No forged-cbm checkpoints found.")
        
    forged_files.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)), reverse=True)
    best_ckpt = forged_files[0]
    ckpt_path = os.path.join(CHECKPOINT_DIR, best_ckpt)
    
    system = CBM_System.load_from_checkpoint(ckpt_path, model=base_model, config=Config)

    print("Freezing Concept Heads to preserve XAI geometry...")
    frozen_count = 0
    for name, param in system.model.named_parameters():
        if 'concept' in name.lower():
            param.requires_grad = False
            frozen_count += 1

    print(f"Ingesting HAM10000 Dataset...")
    full_df = pd.read_csv(META_CSV)
    clean_dx = full_df['dx'].astype(str).str.lower().str.strip()
    binary_labels = (clean_dx == 'mel').astype(int)
    
    train_df, val_df = train_test_split(full_df, test_size=0.2, stratify=binary_labels, random_state=42)

    train_labels = (train_df['dx'].astype(str).str.lower().str.strip() == 'mel').astype(int)
    count_melanoma = train_labels.sum()
    count_other = len(train_df) - count_melanoma
    
    weight_melanoma = 1.0 / count_melanoma if count_melanoma > 0 else 1.0
    weight_other = 1.0 / count_other if count_other > 0 else 1.0
    sample_weights = [weight_melanoma if l == 1 else weight_other for l in train_labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights), 
        num_samples=len(sample_weights), 
        replacement=True
    )

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    train_dataset = HAM10000Dataset(metadata_df=train_df, image_dir=HAM_DIR, transforms=train_transform)
    val_dataset = HAM10000Dataset(metadata_df=val_df, image_dir=HAM_DIR, transforms=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=2) 
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="ham-gated-cbm-{epoch:02d}-{val_auc:.3f}",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_auc",
        min_delta=0.005,
        patience=10,
        verbose=True,
        mode="max"
    )

    logger = TensorBoardLogger("/content/drive/MyDrive/XAI_Cloud_Run/tb_logs", name="GatedHybridCBM_HAM10000")

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",     
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision="16-mixed",   
        log_every_n_steps=10
    )

    # ---------------------------------------------------------
    # THE RESUMPTION OVERRIDE
    # ---------------------------------------------------------
    ham_files = [f for f in os.listdir(CHECKPOINT_DIR) if "ham-gated-cbm" in f and f.endswith(".ckpt")]
    if ham_files:
        # Sort to find the absolute latest interrupted checkpoint
        ham_files.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)), reverse=True)
        resume_ckpt = os.path.join(CHECKPOINT_DIR, ham_files[0])
        print(f"Checkpoint detected. Resuming from exact state: {ham_files[0]}")
        trainer.fit(system, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_ckpt)
    else:
        print("Launching HAM10000 Transfer Training...")
        trainer.fit(system, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
    print(f"Master Architecture Complete. Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
