# Parameter-Efficient Fine-Tuning for Sentiment Classification

[![CI](https://github.com/mnoumanhanif/peft-sentiment-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/mnoumanhanif/peft-sentiment-classification/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

A comprehensive ablation study comparing **Parameter-Efficient Fine-Tuning (PEFT)** methods against Full Fine-Tuning for sentiment classification on the IMDb dataset.

## Overview

This project evaluates **Full Fine-Tuning** vs. three state-of-the-art PEFT methods:

| Method | Description | Trainable Params |
|--------|-------------|-----------------|
| **Full Fine-Tuning** | Updates all model parameters (baseline) | 100% |
| **LoRA** | Low-rank adaptation of attention/FFN layers | ~1-3% |
| **QLoRA** | 4-bit quantized LoRA (NF4) | ~1-3% |
| **IA3** | Learned activation scaling vectors | ~0.03% |

across three transformer architectures:

- **DistilBERT** (67M params)
- **RoBERTa-Base** (125M params)
- **DeBERTa-v3** (184M params)

The study performs a **42-run experimental matrix** (3 architectures x 7 configurations x 2 seeds).

## Key Findings

- **QLoRA** achieves 97-99% of Full FT accuracy with 84-90% VRAM reduction
- **FFN layer placement** is critical - Attention+FFN outperforms Attention-only
- **IA3** is unstable on larger models (55% accuracy on DeBERTa-v3 at LR=1e-5)
- **Learning rate 2e-5** consistently improves all PEFT methods

## Tech Stack

- **PyTorch** - deep learning framework
- **HuggingFace Transformers** - pretrained models and training utilities
- **PEFT** - parameter-efficient fine-tuning library
- **bitsandbytes** - 4-bit quantization for QLoRA
- **datasets** - dataset loading and preprocessing
- **evaluate** - metrics computation
- **matplotlib / seaborn** - visualization

## Project Structure

```
peft-sentiment-classification/
├── src/                    # Reusable Python modules
│   ├── __init__.py
│   ├── config.py           # Experiment configurations and defaults
│   ├── data.py             # Data loading and preprocessing
│   ├── models.py           # Model loading for all fine-tuning methods
│   └── training.py         # Training loop and metrics collection
├── notebooks/              # Jupyter notebooks
│   └── peft_sentiment_classification.ipynb
├── docs/                   # Documentation
│   ├── setup.md            # Installation and setup guide
│   ├── architecture.md     # System architecture overview
│   ├── development.md      # Development workflow guide
│   └── research_paper.pdf  # Full IEEE-format research paper
├── tests/                  # Unit tests
│   └── test_config.py
├── configs/                # Experiment configuration files
│   └── experiment_matrix.json
├── .github/                # CI and GitHub templates
│   ├── workflows/ci.yml
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── requirements.txt
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (4 GB+ VRAM; 16 GB recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/mnoumanhanif/peft-sentiment-classification.git
cd peft-sentiment-classification

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Notebook

```bash
jupyter notebook notebooks/peft_sentiment_classification.ipynb
```

Run all cells sequentially. The full 42-run experiment takes approximately 4-6 hours on an NVIDIA RTX A4000 (16 GB VRAM).

### Using the Python Modules

```python
from src.data import load_tokenizer, load_imdb_dataset, preprocess_dataset, create_metrics_function
from src.models import load_lora_model
from src.training import create_training_args, run_experiment
from src.config import MODELS, get_lora_config

# Prepare data
tokenizer = load_tokenizer(MODELS["distilbert"])
dataset = load_imdb_dataset()
tokenized = preprocess_dataset(dataset, tokenizer)
compute_metrics = create_metrics_function()

# Configure LoRA
lora_cfg = get_lora_config("distilbert", rank=8)
model = load_lora_model(MODELS["distilbert"], lora_cfg)

# Train and evaluate
args = create_training_args("./results/lora_test", learning_rate=2e-5, seed=42)
results = run_experiment(
    model, tokenizer, tokenized["train"], tokenized["test"],
    compute_metrics, args, experiment_name="LoRA DistilBERT"
)
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Linting

```bash
flake8 src/ tests/ --max-line-length=120
```

See [docs/development.md](docs/development.md) for the full development guide.

## Dataset

**IMDb Sentiment Classification** - 25,000 training + 25,000 test samples, binary classification (positive/negative), tokenized to max length 256.

## Experimental Design

| Parameter | Value |
|-----------|-------|
| Epochs | 2 |
| Effective batch size | 16 (8 x 2 gradient accumulation) |
| Learning rates | 1e-5, 2e-5 |
| Seeds | 42, 43 |
| LoRA ranks tested | 8, 16 |
| Hardware | NVIDIA RTX A4000 (16 GB), Intel i7-13700, 32 GB RAM |

## Documentation

- [Setup Guide](docs/setup.md) - installation and environment setup
- [Architecture Overview](docs/architecture.md) - system design and method descriptions
- [Development Guide](docs/development.md) - contributing workflow and code organization
- [Research Paper](docs/research_paper.pdf) - full IEEE-format paper with results and analysis

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```
M. N. Hanif. Parameter-Efficient Fine-Tuning for Sentiment Classification:
A Comprehensive Ablation Study Across Three Architectures. 2025.
```

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Muhammad Nouman Hanif** - [GitHub](https://github.com/mnoumanhanif)

## Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PEFT Library](https://huggingface.co/docs/peft/)
- [IMDb Dataset](https://huggingface.co/datasets/mteb/imdb)
- FAST-NUCES Data Science Department
