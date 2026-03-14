# Development Guide

This guide covers development workflows for contributing to the project.

## Project Structure

```
peft-sentiment-classification/
├── src/                    # Reusable Python modules
│   ├── __init__.py
│   ├── config.py           # Experiment configurations
│   ├── data.py             # Data loading and preprocessing
│   ├── models.py           # Model loading utilities
│   └── training.py         # Training loop and metrics
├── notebooks/              # Jupyter notebooks
│   └── peft_sentiment_classification.ipynb
├── docs/                   # Documentation
│   ├── setup.md
│   ├── architecture.md
│   └── development.md
├── tests/                  # Unit tests
│   └── test_config.py
├── configs/                # Configuration files
│   └── experiment_matrix.json
├── .github/                # GitHub templates and CI
│   ├── workflows/ci.yml
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
└── LICENSE
```

## Development Workflow

### 1. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Make Changes

- **Source modules** are in `src/` — these contain reusable functions extracted from the notebook.
- **Notebook** is in `notebooks/` — this is the primary experiment runner.
- **Tests** are in `tests/` — add tests for any new functionality.

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Lint Code

```bash
# Check for errors
flake8 src/ tests/ --select=E9,F63,F7,F82 --show-source

# Full style check
flake8 src/ tests/ --max-line-length=120
```

### 5. Submit Changes

Follow the [Contributing Guide](../CONTRIBUTING.md) for PR submission.

## Adding a New Model Architecture

1. Add the model checkpoint to `MODELS` in `src/config.py`.
2. Add the target modules for LoRA and IA3 in the corresponding dictionaries.
3. Add a new section in the notebook or create a new experiment script.
4. Update documentation to reflect the new architecture.

## Adding a New Fine-Tuning Method

1. Add a model loader function in `src/models.py`.
2. Add configuration defaults in `src/config.py`.
3. Add experiments to the notebook.
4. Add tests for the new configuration.

## Running Individual Experiments

Use the source modules for quick experiments without the full notebook:

```python
from src.config import MODELS, get_lora_config
from src.data import load_tokenizer, load_imdb_dataset, preprocess_dataset
from src.models import load_lora_model

model_name = "distilbert"
tokenizer = load_tokenizer(MODELS[model_name])
dataset = load_imdb_dataset()
tokenized = preprocess_dataset(dataset, tokenizer, max_length=256)

lora_cfg = get_lora_config(model_name, rank=8)
model = load_lora_model(MODELS[model_name], lora_cfg)
```
