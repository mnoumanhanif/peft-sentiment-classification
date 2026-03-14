"""Experiment configuration definitions.

Provides default configurations for all supported architectures and fine-tuning methods.
"""

# Supported model architectures
MODELS = {
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
    "deberta": "microsoft/deberta-v3-base",
}

# LoRA target modules per architecture
LORA_TARGET_MODULES = {
    "distilbert": ["q_lin", "v_lin"],
    "roberta": ["query", "value"],
    "deberta": ["query_proj", "value_proj"],
}

# LoRA + FFN target modules per architecture
LORA_FFN_TARGET_MODULES = {
    "distilbert": ["q_lin", "v_lin", "ffn.lin1", "ffn.lin2"],
    "roberta": ["query", "value", "intermediate.dense", "output.dense"],
    "deberta": ["query_proj", "value_proj", "intermediate.dense", "output.dense"],
}

# IA3 target modules per architecture
IA3_TARGET_MODULES = {
    "distilbert": ["q_lin", "v_lin", "out_lin"],
    "roberta": ["query", "value", "output.dense"],
    "deberta": ["query_proj", "value_proj", "output.dense"],
}

# IA3 feedforward modules per architecture
IA3_FF_MODULES = {
    "distilbert": ["out_lin"],
    "roberta": ["output.dense"],
    "deberta": ["output.dense"],
}

# Default training hyperparameters
DEFAULT_TRAINING = {
    "num_epochs": 2,
    "per_device_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "max_length": 256,
    "seeds": [42, 43],
    "learning_rates": [1e-5, 2e-5],
}

# Default LoRA hyperparameters
DEFAULT_LORA = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "task_type": "SEQ_CLS",
}

# LoRA rank sensitivity values
LORA_RANK_SENSITIVITY = [8, 16]


def get_lora_config(architecture, rank=8, include_ffn=False):
    """Get LoRA configuration for a given architecture.

    Args:
        architecture: One of 'distilbert', 'roberta', 'deberta'.
        rank: LoRA rank (default: 8).
        include_ffn: Whether to include FFN layers as targets.

    Returns:
        Dictionary with LoRA configuration parameters.
    """
    if architecture not in MODELS:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(MODELS.keys())}")

    modules = LORA_FFN_TARGET_MODULES if include_ffn else LORA_TARGET_MODULES

    config = DEFAULT_LORA.copy()
    config["r"] = rank
    config["target_modules"] = modules[architecture]
    return config


def get_ia3_config(architecture):
    """Get IA3 configuration for a given architecture.

    Args:
        architecture: One of 'distilbert', 'roberta', 'deberta'.

    Returns:
        Dictionary with IA3 configuration parameters.
    """
    if architecture not in MODELS:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(MODELS.keys())}")

    return {
        "target_modules": IA3_TARGET_MODULES[architecture],
        "feedforward_modules": IA3_FF_MODULES[architecture],
        "task_type": "SEQ_CLS",
    }
