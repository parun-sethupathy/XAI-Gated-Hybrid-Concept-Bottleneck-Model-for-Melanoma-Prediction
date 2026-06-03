import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import Config
from models import GatedHybridCBM
from engine import CBM_System

def execute_peak_audit():
    print("INITIALIZING PEAK DUAL-COHORT ACID TEST...")

    BASE_DIR = "/content/drive/MyDrive/XAI_Cloud_Run/"
    HAM_DIR = os.path.join(BASE_DIR, "HAM10000")
    ISIC_DIR = os.path.join(BASE_DIR, "isic_2020")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "cloud_checkpoints")
    
    # Locate the newly generated peak checkpoint
    peak_ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('peak-cbm-')]
    if not peak_ckpts:
        print("ERROR: Peak checkpoint not found.")
        return
    CKPT_PATH = os.path.join(CHECKPOINT_DIR, sorted(peak_ckpts)[-1])

    print(f"Injecting Peak Memory Overlay: {os.path.basename(CKPT_PATH)}...")
    base_model = GatedHybridCBM(
        concept_dims=Config.get_active_concept_dims(),
        backbone=Config.BACKBONE,
        feat_proj_dim=Config.FEAT_PROJ_DIM,
        hidden=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT
    )
    
    system = CBM_System.load_from_checkpoint(CKPT_PATH, model=base_model, config=Config)
    system.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    print("\nPrepping Cohort A: HAM10000 Validation Sample...")
    ham_meta = [f for f in os.listdir(HAM_DIR) if f.endswith('.csv')][0]
    ham_df = pd.read_csv(os.path.join(HAM_DIR, ham_meta))
    
    binary_labels = (ham_df['dx'].astype(str).str.lower().str.strip() == 'mel').astype(int)
    _, val_df = train_test_split(ham_df, test_size=0.2, stratify=binary_labels, random_state=42)
    
    ham_sample = val_df.sample(n=200, random_state=42).reset_index(drop=True)
    ham_results = {'y_true': [], 'y_prob': [], 'alpha': []}
    
    with torch.no_grad():
        for _, row in tqdm(ham_sample.iterrows(), total=len(ham_sample), desc="Auditing HAM10k"):
            img_id = str(row['image_id'])
            img_path = None
            for root, _, files in os.walk(HAM_DIR):
                if f"{img_id}.jpg" in files:
                    img_path = os.path.join(root, f"{img_id}.jpg")
                    break
            
            if img_path:
                img_tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                _, y_logit, alpha = system(img_tensor)
                ham_results['y_true'].append(1 if str(row['dx']).lower().strip() == 'mel' else 0)
                ham_results['y_prob'].append(torch.sigmoid(y_logit).item())
                ham_results['alpha'].append(alpha.item())

    print("\nPrepping Cohort B: ISIC 2020 Micro-Batch...")
    isic_csv = os.path.join(ISIC_DIR, "train.csv")
    isic_img_dir = os.path.join(ISIC_DIR, "images")
    
    if os.path.exists(isic_csv) and os.path.exists(isic_img_dir):
        isic_df = pd.read_csv(isic_csv)
        uploaded_files = [f for f in os.listdir(isic_img_dir) if f.endswith(('.jpg', '.jpeg'))]
        uploaded_basenames = [os.path.splitext(f)[0] for f in uploaded_files]
        test_df = isic_df[isic_df['image_name'].isin(uploaded_basenames)].reset_index(drop=True)
        
        isic_results = {'y_true': [], 'y_prob': [], 'alpha': []}
        
        with torch.no_grad():
            for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Auditing ISIC"):
                img_path = os.path.join(isic_img_dir, str(row['image_name']) + ".jpg")
                img_tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                _, y_logit, alpha = system(img_tensor)
                isic_results['y_true'].append(row['target'])
                isic_results['y_prob'].append(torch.sigmoid(y_logit).item())
                isic_results['alpha'].append(alpha.item())
    else:
        print("WARNING: ISIC Data not found. Skipping Cohort B.")
        isic_results = None

    def print_report(name, res):
        print(f"\n" + "="*50)
        print(f"{name} AUDIT REPORT (PEAK FORM)")
        print("="*50)
        
        y_true = res['y_true']
        y_prob = res['y_prob']
        y_pred = [1 if p > 0.5 else 0 for p in y_prob]
        alphas = res['alpha']
        
        if len(set(y_true)) > 1:
            print(f"AUC Score: {roc_auc_score(y_true, y_prob):.3f}")
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0,0,0,0)
        
        print(f"\n--- NATIVE METRICS (Threshold 0.5) ---")
        print(f"Total Melanomas Present: {tp + fn}")
        print(f"FALSE NEGATIVES (Missed Cancers): {fn}")
        print(f"Sensitivity (Recall): {tp/(tp+fn) if (tp+fn)>0 else 0:.2f}")
        
        print(f"\n--- XAI INTEGRITY ---")
        print(f"Average Alpha Gate: {np.mean(alphas):.3f}")
        print(f"Black-Box Bias (<0.2): {sum(1 for a in alphas if a < 0.2) / len(alphas) * 100:.1f}%")
        
    print_report("COHORT A: HAM10000", ham_results)
    if isic_results:
        print_report("COHORT B: ISIC 2020", isic_results)

if __name__ == "__main__":
    execute_peak_audit()
