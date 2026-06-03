# XAI-Gated Hybrid Concept Bottleneck Model for Melanoma Prediction

> An Explainable AI framework for interpretable melanoma diagnosis using Concept Bottleneck Models, gated concept reasoning, and multi-dataset evaluation.

## Overview

Early detection of melanoma remains one of the most important challenges in medical imaging. While modern deep learning models achieve impressive diagnostic performance, they often operate as black-box systems, providing little insight into *why* a prediction was made.

This project investigates a hybrid approach combining:

* Explainable Artificial Intelligence (XAI)
* Concept Bottleneck Models (CBMs)
* Gated Concept Reasoning
* Dermatology Image Classification

The objective is to build a clinically interpretable diagnostic framework capable of predicting melanoma while exposing the intermediate medical concepts that contribute to each decision.

---

## Motivation

Traditional CNN-based diagnostic systems directly map images to predictions:

```text id="s9jlwm"
Image → Neural Network → Diagnosis
```

Although effective, this provides minimal transparency.

This framework introduces an intermediate concept layer:

```text id="x6k7m3"
Image
  ↓
Concept Extractor
  ↓
Dermatological Concepts
  ↓
Gated Concept Reasoning
  ↓
Final Diagnosis
```

This allows the model to explain decisions through clinically meaningful concepts rather than hidden latent representations.

---

## Research Goals

* Improve interpretability of melanoma classification models.
* Investigate concept-level reasoning in medical AI.
* Reduce reliance on black-box feature representations.
* Compare performance across multiple dermatology datasets.
* Explore gated concept selection mechanisms for improved robustness.
* Evaluate the trade-off between explainability and predictive performance.

---

## Core Framework Components

### Concept Bottleneck Layer

The framework explicitly predicts medically relevant concepts before producing a diagnosis.

Examples include:

* Asymmetry
* Border Irregularity
* Color Variation
* Diameter Characteristics
* Structural Patterns

These concepts form an interpretable representation of the lesion.

---

### Gated Concept Reasoning

Not all concepts contribute equally to every diagnosis.

A gating mechanism dynamically weights concept importance, allowing the model to focus on the most diagnostically relevant information for a given image.

Benefits:

* Improved interpretability
* Reduced concept noise
* Better concept utilization
* Enhanced robustness across datasets

---

### Explainability Module

The framework incorporates multiple explainability approaches including:

* Concept Attribution
* Attention-Based Analysis
* Gradient-Based Explanations
* Feature Importance Mapping

The goal is to provide both prediction confidence and reasoning transparency.

---

## Supported Datasets

Current experiments are conducted using multiple dermatology datasets including:

* HAM10000
* ISIC
* Derm7pt

The framework is designed to support additional dermatological datasets through a modular training pipeline.

---

## Repository Structure

```text id="7yq2z8"
Framework_for_HAM10000/
Framework_for_ISIC/
Framework_for_derm7pt/

configs/
models/
training/
evaluation/
xai/
```

Each dataset pipeline can be trained and evaluated independently while sharing common concept bottleneck and explainability components.

---

## Pretrained Weights Availability

The final trained checkpoints are not included in this repository because they exceed GitHub's file size limits.

Researchers, reviewers, and students interested in reproducing the reported results may request access to the pretrained model weights by contacting:

**parunms@gmail.com**

Please include your intended use case, institution, or research context when requesting access.



## Current Status

Project Status:

```text id="8z9m2q"
Research & Development (Active)
```

Current focus areas:

* Concept bottleneck optimization
* Gated concept fusion
* Cross-dataset generalization
* Explainability benchmarking
* Medical concept validation

---

## Future Work

### Multi-Modal Reasoning

Integrating:

* Clinical metadata
* Patient demographics
* Lesion images

into a unified concept bottleneck framework.

### Concept Verification

Introducing dermatologist-verified concept supervision.

### Advanced Explainability

Integration with:

* SHAP
* LIME
* Integrated Gradients
* Counterfactual Explanations

### Foundation Models

Exploring Vision Transformers and foundation-model backbones while preserving concept-level interpretability.

---

## Research Interests

This project lies at the intersection of:

* Medical AI
* Explainable Artificial Intelligence (XAI)
* Computer Vision
* Trustworthy Machine Learning
* Human-Centered AI
* Concept-Based Learning

---

## Disclaimer

This repository is an active research framework and is intended for educational and research purposes. It is not designed for clinical deployment or medical decision-making without extensive validation and regulatory approval.
