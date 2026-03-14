"""Training utilities and experiment runner."""

import time

import torch
from transformers import TrainingArguments, Trainer

from .models import count_trainable_parameters


def create_training_args(
    output_dir,
    learning_rate=1e-5,
    num_epochs=2,
    per_device_batch_size=8,
    gradient_accumulation_steps=2,
    seed=42,
):
    """Create HuggingFace TrainingArguments for an experiment.

    Args:
        output_dir: Directory to save training outputs.
        learning_rate: Learning rate for optimization.
        num_epochs: Number of training epochs.
        per_device_batch_size: Batch size per device.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        seed: Random seed for reproducibility.

    Returns:
        TrainingArguments instance.
    """
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        logging_dir=f"{output_dir}/logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )


def run_experiment(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    compute_metrics,
    training_args,
    experiment_name="experiment",
):
    """Run a training experiment and collect metrics.

    Args:
        model: Model to train.
        tokenizer: Tokenizer for the model.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        compute_metrics: Function to compute evaluation metrics.
        training_args: TrainingArguments instance.
        experiment_name: Name for logging purposes.

    Returns:
        Dictionary containing experiment results:
            - accuracy: Test accuracy
            - f1_macro: Test F1-macro score
            - training_time_s: Training time in seconds
            - peak_vram_gb: Peak VRAM usage in GB
            - trainable_params: Number of trainable parameters
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainable_params = count_trainable_parameters(model)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"--- Starting: {experiment_name} ---")
    print(f"Trainable Parameters: {trainable_params / 1_000_000:.2f}M")

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    peak_vram_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_vram_gb = peak_vram_bytes / (1024**3)

    print("\n--- Evaluating ---")
    eval_results = trainer.evaluate()

    results = {
        "experiment": experiment_name,
        "accuracy": eval_results["eval_accuracy"],
        "f1_macro": eval_results["eval_f1_macro"],
        "training_time_s": round(training_time, 2),
        "peak_vram_gb": round(peak_vram_gb, 4),
        "trainable_params": trainable_params,
    }

    print(f"\n--- Results: {experiment_name} ---")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-macro: {results['f1_macro']:.4f}")
    print(f"Training Time: {results['training_time_s']}s")
    print(f"Peak VRAM: {results['peak_vram_gb']} GB")

    return results
