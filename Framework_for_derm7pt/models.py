# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MaskedFocalConceptLoss(nn.Module):
    """ Custom loss that safely ignores missing clinical concepts """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets, mask, class_weight=None):
        valid = mask > 0.5
        if valid.sum() == 0: 
            return logits.sum() * 0.0
            
        logits_v, targets_v = logits[valid], targets[valid]
        ce = F.cross_entropy(logits_v, targets_v, weight=class_weight, reduction="none")
        pt = torch.softmax(logits_v, dim=1)[torch.arange(len(targets_v)), targets_v]
        return (((1 - pt) ** self.gamma) * ce).mean()

class BinaryFocalLoss(nn.Module):
    """ Asymmetric loss to heavily penalize missing Melanoma (Recall optimization) """
    def __init__(self, alpha=0.85, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * ((1 - pt) ** self.gamma) * bce).mean()

class GatedHybridCBM(nn.Module):
    """ Dynamic architecture that scales to any dataset configuration """
    def __init__(self, concept_dims, backbone="tf_efficientnetv2_s", feat_proj_dim=256, hidden=256, dropout=0.30):
        super().__init__()
        
        # 1. Visual Feature Extractor
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features

        # 2. Dynamic Concept Heads (Auto-generates based on the config list!)
        self.concept_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2), 
                nn.ReLU(), 
                nn.Dropout(dropout),
                nn.Linear(feat_dim // 2, d)
            ) for d in concept_dims
        ])

        # 3. Visual Bypass Projection
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        total_concept_dim = sum(concept_dims)

        # 4. Standard Image-to-Label Head
        self.image_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

        # 5. Concept-to-Label Head
        self.concept_head = nn.Sequential(
            nn.Linear(total_concept_dim + feat_proj_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

        # 6. The Gating Mechanism (Alpha)
        self.gate = nn.Sequential(
            nn.Linear(total_concept_dim + feat_proj_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feats = self.backbone(x)
        
        # Extract Concepts
        c_logits = [head(feats) for head in self.concept_heads]
        c_probs = torch.cat([torch.softmax(logit, dim=1) for logit in c_logits], dim=1)
        
        # Extract Visuals
        feat_vec = self.feat_proj(feats)
        
        # Calculate Independent Logits
        img_logit = self.image_head(feats).squeeze(1)
        cbm_logit = self.concept_head(torch.cat([c_probs, feat_vec], dim=1)).squeeze(1)
        
        # Gated Integration
        alpha = torch.sigmoid(self.gate(torch.cat([c_probs, feat_vec], dim=1))).squeeze(1)
        y_logit = alpha * cbm_logit + (1.0 - alpha) * img_logit
        
        return c_logits, y_logit, alpha