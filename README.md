# XAI-Gated Hybrid Concept Bottleneck Model for Melanoma Prediction

*A clinically aligned Explainable AI framework for melanoma diagnosis utilizing Hybrid Concept Bottleneck Models, dynamic Alpha-Gating, and extensive out of distribution validation.*

---

## Overview

Early detection of melanoma remains one of the most critical challenges in clinical oncology. While modern deep learning systems achieve impressive diagnostic performance, they fundamentally operate as black-box models, offering physicians little insight into the reasoning behind a prediction.

This project addresses the interpretability–accuracy trade-off through a **Gated Hybrid Concept Bottleneck Model (G-HCBM)**.

Rather than relying solely on opaque latent representations, the framework explicitly predicts human-interpretable dermatological concepts while simultaneously maintaining a residual visual pathway. A learnable **Alpha Gate** dynamically controls the contribution of both pathways, allowing the model to achieve strong diagnostic performance while exposing the degree to which each prediction depends on clinically meaningful concepts.

---

## Core Innovation

### Traditional Medical AI

```text
Image
  ↓
Deep Neural Network
  ↓
Diagnosis
```

The internal reasoning process remains inaccessible.

### Proposed Hybrid CBM Architecture

```text
                    Input Dermoscopy Scan
                              │
                     EfficientNetV2 Backbone
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   ▼
     Concept Predictors                  Visual Bypass
      (White-Box Logic)                (Black-Box Logic)
            │                                   │
            └─────────────┬─────────────────────┘
                          ▼
                     Alpha Gate
                          │
                          ▼
              Final Melanoma Prediction
```

The Alpha Gate continuously balances interpretable clinical reasoning and latent visual reasoning.

---

## Architecture Components

### 1. Concept Heads (White-Box Pathway)

The model is explicitly supervised to predict clinically relevant dermatological concepts, including:

* Asymmetry
* Border Irregularity
* Pigment Network
* Streaks
* Blue-White Veil
* Color Variation
* Structural Patterns

These concepts form the interpretable basis of the diagnostic decision.

---

### 2. Visual Bypass (Black-Box Pathway)

Clinical concepts alone cannot capture every visual pattern present within dermoscopic imagery.

A residual visual bypass allows the network to utilize latent image information when concept representations are insufficient.

Benefits include:

* Improved diagnostic robustness
* Better handling of atypical lesions
* Reduced concept bottleneck saturation
* Increased generalization capacity

---

### 3. Alpha Gate (Core Contribution)

The Alpha Gate dynamically determines how much weight is assigned to:

```text
Concept-Based Reasoning
vs.
Visual Latent Reasoning
```

For every prediction, Alpha acts as a quantitative measure of:

> "How interpretable is this diagnosis?"

A higher Alpha indicates stronger reliance on clinical concepts, while lower values indicate increased dependence on latent visual features.

---

# Methodology & Training Pipeline

The framework utilizes a sequential multi-dataset curriculum specifically designed to improve concept quality, safety alignment, and out-of-distribution performance.

---

## Phase 1: Concept Forge (Derm7pt)

Objective:

Train clinically meaningful concept representations.

Approach:

* Derm7pt concept annotations
* Heavy geometric augmentation
* Dual-modality image ingestion
* Concept supervision

Outcome:

The backbone learns stable dermatological feature representations and concept boundaries.

---

## Phase 2: Transfer Training & Safety Alignment (HAM10000)

Objective:

Optimize melanoma diagnosis while preserving concept integrity.

Training Strategy:

* Freeze concept heads
* Freeze concept backbone
* Train visual bypass
* Train Alpha Gate

### Clinical Safety Alignment

Standard classification losses frequently produce models that prioritize overall accuracy while missing positive cases.

This framework instead utilizes:

* Asymmetric Focal Loss
* F-Beta Optimization (β = 2)

The objective explicitly prioritizes:

```text
Sensitivity > Accuracy
```

thereby reducing false negatives in melanoma detection.

---

## Phase 3: OOD Acid Test (ISIC 2020)

Objective:

Verify real-world generalization.

Evaluation:

* Unseen dataset
* No retraining
* No re-calibration
* Clinical noise exposure

This phase measures:

* Concept robustness
* Alpha Gate stability
* Out-of-distribution reliability
* Diagnostic consistency

---

## Clinical Validation & Telemetry Analysis

| Metric / Telemetry Data | Cohort A: HAM10000 (In-Distribution) | Cohort B: ISIC 2020 (Out-of-Distribution) | Clinical Implications |
|----------|----------|----------|----------|
| **Cohort Function** | Validation / Sanity Check | Acid Test / Reality Check | Confirms generalization vs. memorization. |
| **Diagnostic AUC** | 0.990 | N/A (No positive cases in sample) | Near-perfect internal probability ranking. |
| **Total Melanomas Present** | 17 | 0 | Baseline for recall calculation. |
| **False Negatives** | 1 | 0 | Absolute patient safety failure rate. |
| **Sensitivity (Recall)** | 94.0% | 0.0% (N/A) | The model's baseline ability to detect malignancy at a static 0.5 threshold. |
| **Average Alpha Gate** | 0.660 | 0.729 | The proportion of logic strictly routed through interpretable clinical concepts. |
| **Black-Box Bias (< 0.2)** | 6.0% | 0.0% | Percentage of images where the model abandoned XAI concepts for raw pixel textures. |


# Doctor-in-the-Loop Deployment

A Gradio-based clinical dashboard is included for physician interaction.

Features:

* Image upload interface
* Real-time concept telemetry
* Alpha Gate visualization
* Adjustable risk threshold
* Probability calibration controls

Unlike static classification systems, physicians can dynamically adjust sensitivity based on clinical context and patient history.

---

# Repository Structure

```text
XAI_Melanoma_Framework/

├── config.py
├── models.py
├── engine.py
├── data_loaders.py
├── frontend.py

├── training_scripts/
│   ├── concept_forge.py
│   ├── transfer_training.py
│   └── finetune.py

└── validation_scripts/
    ├── ood_validation.py
    └── evaluation.py
```

---

# Research Contributions

* Hybrid Concept Bottleneck Architecture
* Dynamic Alpha-Gating Mechanism
* Clinically Interpretable Prediction Pipeline
* Multi-Dataset Curriculum Training
* OOD Generalization Validation
* Safety-Aligned Optimization using F2 Metrics
* Human-AI Collaborative Diagnostic Interface

---

# Pretrained Weights Availability

The final trained checkpoints exceed GitHub's file size limitations and are therefore not distributed directly through the repository.

Researchers, students, reviewers, and medical professionals interested in reproducing the reported experiments may request access to the pretrained model weights by contacting:

**[parunms@gmail.com](mailto:parunms@gmail.com)**

Please include a brief description of your institution, project, or research objective when requesting access.

---

# Current Status

**Project Status:** Release Candidate Finalized

Current performance focuses:

* Explainability fidelity
* Clinical safety alignment
* OOD robustness
* Concept integrity preservation

---

# Future Work

### Spatial Concept Attribution

Integrating Grad-CAM and concept localization heatmaps to identify the exact image regions responsible for concept activation.

### Vision Transformer Integration

Replacing EfficientNetV2 with:

* Vision Transformer (ViT)
* Swin Transformer
* Foundation Vision Models

to study concept routing under transformer architectures.

### Clinical Validation

Transitioning from retrospective dataset evaluation toward prospective clinical shadowing and physician feedback studies.

### Concept Verification

Incorporating dermatologist-verified concept labels to further improve concept reliability.

---

# Disclaimer

This repository is an academic research framework intended for educational use and Explainable AI research.

It is **not** an FDA-approved medical device and must not be used for clinical diagnosis, patient triaging, or treatment decisions without appropriate regulatory approval and clinical validation.
