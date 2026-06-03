XAI-Gated Hybrid Concept Bottleneck Model for Melanoma Prediction
A clinically aligned Explainable AI framework for melanoma diagnosis utilizing Hybrid Concept Bottleneck Models, dynamic Alpha-Gating, and rigorous out-of-distribution validation.

Overview
Early detection of melanoma is a critical challenge in clinical oncology. While modern deep learning models achieve impressive Area Under the Curve (AUC) metrics, they inherently operate as black-box systems, providing physicians with little to no insight into why a prediction was made.

This project solves the interpretability-accuracy trade-off by introducing a Gated Hybrid Concept Bottleneck Model.

Instead of relying solely on opaque latent spaces, this framework forces the neural network to output human-readable dermatological concepts (e.g., Pigment Networks, Streaks) alongside a raw visual bypass. A dynamic "Alpha Gate" routes the logic, allowing the model to achieve peak diagnostic performance while mathematically proving its reliance on interpretable clinical features.

Theoretical Architecture
Traditional diagnostic systems utilize a direct mapping approach:

Plaintext
Image -> Deep Neural Network -> Diagnosis (Black-Box)
This framework introduces a dual-pathway Hybrid CBM:

Plaintext
                  Input Dermoscopy Scan
                            |
                 EfficientNetV2 Backbone
                            |
            |-------------------------------|
            v                               v
    Concept Predictors                Visual Bypass
    (White-Box Logic)               (Black-Box Logic)
            |                               |
            |------>  The Alpha Gate <------|
                            |
                 Final Melanoma Prediction
1. The Concept Heads (White-Box)
The model is explicitly supervised to detect seven foundational dermatological criteria (e.g., Asymmetry, Border Irregularity, Blue/White Veil). These form the interpretable basis of the prediction.

2. The Visual Bypass (Black-Box)
Because rigid clinical concepts cannot capture 100% of the variance in raw medical data, a residual visual bypass is permitted. This allows the model to utilize uninterpretable pixel data for edge-cases that lack clear morphological concept signatures.

3. The Alpha Gate
The core innovation of this framework. The Alpha Gate is a learned parameter that dynamically controls the fusion of the White-Box and Black-Box pathways. By monitoring the Alpha value, physicians can immediately quantify the "XAI Integrity" of any given diagnosis (i.e., exactly how much the model relied on readable concepts vs. opaque pixel textures).

Methodology & Training Pipeline
This framework abandons standard single-dataset training in favor of a sequential, multi-dataset curriculum designed to prevent mode collapse and ensure out-of-distribution (OOD) generalization.

Phase 1: The Concept Forge (Derm7pt)
The network's concept heads are initially trained on the Derm7pt dataset, utilizing heavy geometric augmentation and dual-modality ingestion (clinical + dermoscopic images) to force the backbone to learn strict structural boundaries for clinical concepts.

Phase 2: Transfer Training & Safety Alignment (HAM10000)
The visual backbone and concept heads are frozen. The model is then exposed to the large-scale HAM10000 dataset to train the visual bypass and the Alpha Gate.

Clinical Safety Alignment: Standard accuracy metrics create conservative models that miss positive cases. This framework utilizes an asymmetric Focal Loss and FBeta (F2) optimization to aggressively penalize False Negatives, intentionally shifting the decision boundary to prioritize patient safety (Sensitivity) over raw Accuracy.

Phase 3: The OOD Acid Test (ISIC 2020)
The locked architecture is evaluated on a completely unseen, uncurated micro-batch from the ISIC 2020 challenge to verify that the visual feature maps and XAI routing logic generalize to real-world clinical noise.

Clinical Deployment: Doctor-in-the-Loop
This repository includes a Gradio-based clinical dashboard (frontend.py) designed for physician interaction.

Rather than enforcing a static 0.5 probability threshold, the software provides an Interactive Risk Threshold. This allows physicians to dynamically adjust the model's sensitivity based on patient history, viewing real-time readouts of the Alpha Gate telemetry and concept activation confidence.

Repository Structure
Plaintext
XAI_Melanoma_Framework/
├── config.py                 # Hyperparameters and structural dimensions
├── models.py                 # Core GatedHybridCBM PyTorch architecture
├── engine.py                 # LightningModule for training logic and loss metrics
├── data_loaders.py           # Multi-dataset ingestion and transform pipelines
├── frontend.py               # Gradio dashboard for clinical inference
├── training_scripts/         # Sequential scripts (Forge, Transfer, Finetune)
└── validation_scripts/       # OOD Acid Testing and metric evaluation

Pretrained Weights Availability
Due to file size constraints, the fully optimized peak-cbm checkpoints (containing the frozen BatchNorm states and F2-aligned safety metrics) are not hosted directly in this repository.

Researchers and medical professionals interested in reproducing the OOD validation results or testing the clinical dashboard may request access to the .ckpt files by contacting:

parunms@gmail.com

Please include a brief description of your intended research context or institution when requesting access.

Current Status & Future Work
Project Status: Release Candidate Finalized.

Future Objectives:

Spatial Activation Mapping: Upgrading the concept telemetry to output Grad-CAM heatmaps, showing exactly where on the image the concept heads detect features like "Streaks."

Foundation Backbone Migration: Replacing the EfficientNetV2 extractor with Vision Transformer (ViT) architectures to evaluate cross-attention concept routing.

Clinical Trial Validation: Moving beyond retrospective dataset evaluation to prospective clinical shadowing.

Disclaimer
This repository is an active academic research framework. It is intended strictly for educational purposes and algorithmic research in the field of Explainable AI. It is not an FDA-approved medical device and must not be used for clinical decision-making, patient diagnosis, or triaging.
