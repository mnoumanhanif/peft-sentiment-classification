"""Model loading utilities for different fine-tuning methods."""

import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, IA3Config, prepare_model_for_kbit_training


def load_base_model(model_checkpoint, num_labels=2):
    """Load a base model for sequence classification.

    Args:
        model_checkpoint: HuggingFace model identifier.
        num_labels: Number of output labels (default: 2 for binary classification).

    Returns:
        AutoModelForSequenceClassification instance.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels
    )


def load_lora_model(model_checkpoint, lora_config, num_labels=2):
    """Load a model with LoRA adapters applied.

    Args:
        model_checkpoint: HuggingFace model identifier.
        lora_config: Dictionary with LoRA configuration parameters.
            Expected keys: r, lora_alpha, lora_dropout, target_modules, task_type.
        num_labels: Number of output labels.

    Returns:
        PEFT model with LoRA adapters.
    """
    base_model = load_base_model(model_checkpoint, num_labels)

    peft_config = LoraConfig(
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        target_modules=lora_config.get("target_modules", ["q_lin", "v_lin"]),
        task_type=lora_config.get("task_type", "SEQ_CLS"),
    )

    model = get_peft_model(base_model, peft_config)
    return model


def load_qlora_model(model_checkpoint, lora_config, num_labels=2):
    """Load a 4-bit quantized model with LoRA adapters (QLoRA).

    Args:
        model_checkpoint: HuggingFace model identifier.
        lora_config: Dictionary with LoRA configuration parameters.
        num_labels: Number of output labels.

    Returns:
        PEFT model with QLoRA configuration.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels, quantization_config=bnb_config
    )

    base_model = prepare_model_for_kbit_training(base_model)

    peft_config = LoraConfig(
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        target_modules=lora_config.get("target_modules", ["q_lin", "v_lin"]),
        task_type=lora_config.get("task_type", "SEQ_CLS"),
    )

    model = get_peft_model(base_model, peft_config)
    return model


def load_ia3_model(model_checkpoint, ia3_config, num_labels=2):
    """Load a model with IA3 adapters applied.

    Args:
        model_checkpoint: HuggingFace model identifier.
        ia3_config: Dictionary with IA3 configuration parameters.
            Expected keys: target_modules, feedforward_modules, task_type.
        num_labels: Number of output labels.

    Returns:
        PEFT model with IA3 adapters.
    """
    base_model = load_base_model(model_checkpoint, num_labels)

    peft_config = IA3Config(
        target_modules=ia3_config.get("target_modules", ["q_lin", "v_lin", "out_lin"]),
        feedforward_modules=ia3_config.get("feedforward_modules", ["out_lin"]),
        task_type=ia3_config.get("task_type", "SEQ_CLS"),
    )

    model = get_peft_model(base_model, peft_config)
    return model


def count_trainable_parameters(model):
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Integer count of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
