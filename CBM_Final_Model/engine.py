# engine.py
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, AUROC, Precision, Recall
from models import MaskedFocalConceptLoss, BinaryFocalLoss

class CBM_System(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        
        self.concept_loss_fn = MaskedFocalConceptLoss(gamma=config.FOCAL_GAMMA)
        self.task_loss_fn = BinaryFocalLoss(alpha=config.FOCAL_ALPHA_MELANOMA, gamma=config.FOCAL_GAMMA)
        
        self.val_acc = Accuracy(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_auc = AUROC(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")

    def on_train_start(self):
        # Mitigation 1: Concept Warm-Up
        # Freeze the final diagnostic heads and visual bypass. 
        # The backbone MUST learn clinical concepts first.
        for param in self.model.image_head.parameters():
            param.requires_grad = False
        for param in self.model.concept_head.parameters():
            param.requires_grad = False
        for param in self.model.gate.parameters():
            param.requires_grad = False
        print("\nPhase 1: Diagnostic Heads Frozen. Forcing Backbone Concept Extraction...")

    def on_train_epoch_start(self):
        # Unfreeze everything at Epoch 5 for Joint Gated Training
        if self.current_epoch == 5:
            for param in self.model.parameters():
                param.requires_grad = True
            print("\nPhase 2: All components unfrozen. Initiating Joint Gated Training...")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, concepts, masks, label = batch
        c_logits, y_logit, alpha = self(img)
        
        c_loss = sum([self.concept_loss_fn(c_logits[i], concepts[:, i], masks[:, i]) for i in range(len(c_logits))])
        t_loss = self.task_loss_fn(y_logit, label)
        
        # Mitigation 2: Dynamic Lambda Annealing
        # Concepts are heavily prioritized early, shifting to Task diagnosis in later epochs
        progress = self.current_epoch / self.trainer.max_epochs
        dynamic_lambda_concept = self.config.LAMBDA_CONCEPT * (1.0 - progress)
        dynamic_lambda_task = self.config.LAMBDA_TASK * (0.5 + progress)
        
        total_loss = (dynamic_lambda_concept * c_loss) + (dynamic_lambda_task * t_loss)
        
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        img, concepts, masks, label = batch
        c_logits, y_logit, alpha = self(img)
        
        c_loss = sum([self.concept_loss_fn(c_logits[i], concepts[:, i], masks[:, i]) for i in range(len(c_logits))])
        t_loss = self.task_loss_fn(y_logit, label)
        
        # Validation uses static config lambdas for consistent metric tracking
        total_loss = (self.config.LAMBDA_CONCEPT * c_loss) + (self.config.LAMBDA_TASK * t_loss)
        
        preds = torch.sigmoid(y_logit)
        self.val_acc.update(preds, label)
        self.val_f1.update(preds, label)
        self.val_auc.update(preds, label)
        self.val_precision.update(preds, label)
        self.val_recall.update(preds, label)
        
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_auc", self.val_auc, on_epoch=True, prog_bar=True)
        self.log("val_recall", self.val_recall, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.LEARNING_RATE, 
            weight_decay=self.config.WEIGHT_DECAY
        )
        return optimizer