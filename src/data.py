"""Data loading and preprocessing utilities for sentiment classification."""

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer


def load_tokenizer(model_checkpoint):
    """Load a pretrained tokenizer for the given model checkpoint.

    Args:
        model_checkpoint: HuggingFace model identifier (e.g., 'distilbert-base-uncased').

    Returns:
        AutoTokenizer instance.
    """
    return AutoTokenizer.from_pretrained(model_checkpoint)


def load_imdb_dataset():
    """Load the IMDb sentiment classification dataset.

    Returns:
        HuggingFace DatasetDict with 'train' and 'test' splits.
    """
    return load_dataset("mteb/imdb")


def preprocess_dataset(dataset, tokenizer, max_length=256):
    """Tokenize and format dataset for training.

    Args:
        dataset: HuggingFace DatasetDict.
        tokenizer: Tokenizer to use for encoding text.
        max_length: Maximum token sequence length (default: 256).

    Returns:
        Tokenized DatasetDict formatted for PyTorch.
    """

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    return tokenized_dataset


def create_metrics_function():
    """Create a compute_metrics function for the HuggingFace Trainer.

    Returns:
        Function that computes accuracy and F1-macro from model predictions.
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = accuracy_metric.compute(
            predictions=predictions, references=labels
        )["accuracy"]

        f1 = f1_metric.compute(
            predictions=predictions, references=labels, average="macro"
        )["f1"]

        return {
            "accuracy": accuracy,
            "f1_macro": f1,
        }

    return compute_metrics
