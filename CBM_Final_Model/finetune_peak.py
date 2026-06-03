import os
import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchmetrics.classification import BinaryFBetaScore, BinaryRecall

from config import Config
from data_loaders import HAM10000Dataset
from models import GatedHybridCBM
from engine import CBM_System

# ---------------------------------------------------------
# PEAK OVERRIDE ENGINE: Native Metric Maximization
# ---------------------------------------------------------
class Finetune_Peak_System(CBM_System):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.val_f2 = BinaryFBetaScore(beta=2.0)
        self.val_recall = BinaryRecall()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        # CRITICAL FIX: Force the entire architecture into eval mode.
        # This permanently locks BatchNorm running stats.
        # The optimizer will still update the weights that have requires_grad=True.
        self.model.eval()

    def training_step(self, batch, batch_idx):
        x, c_true, c_mask, y_true = batch
        c_logits, y_logit, alpha = self.model(x)

        # 5x Native Penalty for False Negatives
        task_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=self.device))
        loss = task_loss_fn(y_logit.view(-1), y_true.float().view(-1))
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, c_true, c_mask, y_true = batch
        c_logits, y_logit, alpha = self.model(x)

        y_prob = torch.sigmoid(y_logit).view(-1)
        y_true = y_true.view(-1)

        self.val_f2(y_prob, y_true)
        self.val_recall(y_prob, y_true)

        self.log("val_f2", self.val_f2, on_epoch=True, prog_bar=True)
        self.log("val_recall", self.val_recall, on_epoch=True, prog_bar=True)
        
        task_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=self.device))
        val_loss = task_loss_fn(y_logit.view(-1), y_true.float().view(-1))
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        return torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4) 

def main():
    pl.seed_everything(42)
    print("INITIALIZING PEAK FORM: State Override & Native Boundary Shift...")

    HAM_DIR = "/content/drive/MyDrive/XAI_Cloud_Run/HAM10000"
    CHECKPOINT_DIR = "/content/drive/MyDrive/XAI_Cloud_Run/cloud_checkpoints/"
    
    meta_files = [f for f in os.listdir(HAM_DIR) if f.endswith('.csv')]
    META_CSV = os.path.join(HAM_DIR, meta_files[0])

    print("Retrieving PRISTINE Release Candidate (Epoch 20)...")
    base_model = GatedHybridCBM(
        concept_dims=Config.get_active_concept_dims(),
        backbone=Config.BACKBONE,
        feat_proj_dim=Config.FEAT_PROJ_DIM,
        hidden=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT
    )
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, "ham-gated-cbm-epoch=20-val_auc=0.950.ckpt")
    base_system = CBM_System.load_from_checkpoint(ckpt_path, model=base_model, config=Config)
    system = Finetune_Peak_System(model=base_system.model, config=Config)

    print("Executing Bulletproof Exclusion Freeze...")
    frozen_count = 0
    trainable_count = 0
    for name, param in system.model.named_parameters():
        if 'backbone' in name.lower() or 'concept' in name.lower() or 'gate' in name.lower() or 'alpha' in name.lower():
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True
            trainable_count += 1
            
    print(f"Locked {frozen_count} visual/concept/gate tensors.")
    print(f"Unlocked {trainable_count} final classification tensors.")

    full_df = pd.read_csv(META_CSV)
    clean_dx = full_df['dx'].astype(str).str.lower().str.strip()
    binary_labels = (clean_dx == 'mel').astype(int)
    
    train_df, val_df = train_test_split(full_df, test_size=0.2, stratify=binary_labels, random_state=42)

    train_labels = (train_df['dx'].astype(str).str.lower().str.strip() == 'mel').astype(int)
    weight_melanoma = 1.0 / train_labels.sum()
    weight_other = 1.0 / (len(train_df) - train_labels.sum())
    sample_weights = [weight_melanoma if l == 1 else weight_other for l in train_labels]
    
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)

    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    train_dataset = HAM10000Dataset(metadata_df=train_df, image_dir=HAM_DIR, transforms=train_transform)
    val_dataset = HAM10000Dataset(metadata_df=val_df, image_dir=HAM_DIR, transforms=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=2) 
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="peak-cbm-{epoch:02d}-{val_f2:.3f}",
        monitor="val_f2", 
        mode="max",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_f2",
        min_delta=0.005,
        patience=3, 
        verbose=True,
        mode="max"
    )

    logger = TensorBoardLogger("/content/drive/MyDrive/XAI_Cloud_Run/tb_logs", name="GatedHybridCBM_Peak")

    trainer = pl.Trainer(
        max_epochs=10, 
        accelerator="auto",     
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision="16-mixed",   
        log_every_n_steps=10
    )

    print("Executing Native Boundary Shift...")
    trainer.fit(system, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Peak Model finalized. Saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
