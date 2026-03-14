# Architecture Overview

This document describes the architecture and design of the PEFT Sentiment Classification project.

## Project Goal

Compare four fine-tuning strategies for transformer-based sentiment classification on the IMDb dataset, measuring trade-offs between accuracy, VRAM usage, and training speed.

## System Architecture

```
┌──────────────────────────────────────────────────────┐
│                    IMDb Dataset                       │
│              (25K train / 25K test)                   │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│              Data Preprocessing                       │
│         Tokenization (max_length=256)                │
│         Format for PyTorch                           │
└──────────────────┬───────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   DistilBERT  RoBERTa   DeBERTa-v3
     (67M)     (125M)     (184M)
        │          │          │
        ▼          ▼          ▼
┌──────────────────────────────────────────────────────┐
│            Fine-Tuning Methods                        │
│                                                       │
│  ┌─────────┐ ┌──────┐ ┌───────┐ ┌─────┐             │
│  │ Full FT │ │ LoRA │ │ QLoRA │ │ IA3 │             │
│  └─────────┘ └──────┘ └───────┘ └─────┘             │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│           Metrics Collection                          │
│   Accuracy, F1-macro, VRAM, Training Time            │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│         Results Aggregation & Visualization           │
│     Pareto Frontier, Rank Sensitivity Plots          │
└──────────────────────────────────────────────────────┘
```

## Fine-Tuning Methods

### 1. Full Fine-Tuning (Baseline)

Updates **all** model parameters. Provides the highest accuracy but requires the most VRAM and compute.

- **Trainable params**: 100% of model
- **VRAM**: Highest (3–7 GB depending on model)

### 2. LoRA (Low-Rank Adaptation)

Injects rank-decomposed matrices `(B × A)` into selected linear layers. Only the low-rank matrices are trained.

- **Trainable params**: ~1–3% of model
- **Key hyperparameters**: rank `r`, `lora_alpha`, target modules
- **Variants tested**: Attention-only, FFN-only, Attention+FFN, rank 8 vs 16

### 3. QLoRA (Quantized LoRA)

Applies LoRA on top of a 4-bit quantized base model using NF4 quantization.

- **VRAM reduction**: 84–90% compared to Full FT
- **Trade-off**: Slightly slower due to dequantization overhead

### 4. IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

Scales activations via learned vectors. Smallest parameter footprint.

- **Trainable params**: ~0.03% of model
- **Limitation**: Unstable convergence on larger models (DeBERTa-v3)

## Code Modules

| Module | Purpose |
|--------|---------|
| `src/data.py` | Dataset loading, tokenization, metrics setup |
| `src/models.py` | Model loading for all fine-tuning methods |
| `src/training.py` | Training loop, experiment runner, metrics collection |
| `src/config.py` | Architecture-specific configurations and defaults |

## Experimental Design

The study uses a **42-run experimental matrix**:

- **3 architectures** × **7 configurations** × **2 seeds** = 42 runs
- Seeds (42, 43) ensure reproducibility; results are averaged
- Learning rates: 1e-5 and 2e-5
- Epochs: 2
- Effective batch size: 16 (8 per device × 2 gradient accumulation)
