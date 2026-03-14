# Setup Guide

This guide covers how to set up the project for running experiments.

## Prerequisites

- **Python** 3.10 or higher
- **CUDA-capable GPU** with at least 4 GB VRAM (16 GB recommended for full experiments)
- **NVIDIA drivers** and **CUDA toolkit** installed
- **Git** for version control

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mnoumanhanif/peft-sentiment-classification.git
cd peft-sentiment-classification
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

## Running Experiments

### Using the Notebook

```bash
jupyter notebook notebooks/
```

Open `peft_sentiment_classification.ipynb` and run all cells sequentially.

> **Note:** The full 42-run experiment takes 4–6 hours on an NVIDIA RTX A4000 (16 GB VRAM).

### Using the Source Modules

You can also use the extracted Python modules directly:

```python
from src.data import load_tokenizer, load_imdb_dataset, preprocess_dataset, create_metrics_function
from src.models import load_base_model, load_lora_model
from src.training import create_training_args, run_experiment
from src.config import MODELS, get_lora_config

# Load data
tokenizer = load_tokenizer(MODELS["distilbert"])
dataset = load_imdb_dataset()
tokenized = preprocess_dataset(dataset, tokenizer)
compute_metrics = create_metrics_function()

# Configure and run
lora_cfg = get_lora_config("distilbert", rank=8)
model = load_lora_model(MODELS["distilbert"], lora_cfg)
args = create_training_args("./results/lora_test", learning_rate=2e-5, seed=42)

results = run_experiment(
    model, tokenizer, tokenized["train"], tokenized["test"],
    compute_metrics, args, experiment_name="LoRA DistilBERT"
)
```

## Troubleshooting

### CUDA Out of Memory

- Reduce `per_device_batch_size` to 4 or 2.
- Use QLoRA instead of LoRA or Full Fine-Tuning.
- Reduce `max_length` from 256 to 128.

### bitsandbytes Issues

- Ensure you have a compatible CUDA version for `bitsandbytes`.
- On Windows, use `bitsandbytes-windows` or WSL2.

### Slow Training

- Verify GPU is being used: `torch.cuda.is_available()` should return `True`.
- Check that `accelerate` is properly configured.
