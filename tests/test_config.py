"""Tests for the configuration module."""

from src.config import (
    MODELS,
    LORA_TARGET_MODULES,
    LORA_FFN_TARGET_MODULES,
    IA3_TARGET_MODULES,
    IA3_FF_MODULES,
    DEFAULT_TRAINING,
    DEFAULT_LORA,
    LORA_RANK_SENSITIVITY,
    get_lora_config,
    get_ia3_config,
)


class TestModels:
    """Tests for model configuration constants."""

    def test_all_architectures_defined(self):
        assert "distilbert" in MODELS
        assert "roberta" in MODELS
        assert "deberta" in MODELS

    def test_model_checkpoints_are_strings(self):
        for name, checkpoint in MODELS.items():
            assert isinstance(checkpoint, str), f"Checkpoint for {name} is not a string"
            assert len(checkpoint) > 0, f"Checkpoint for {name} is empty"

    def test_three_architectures(self):
        assert len(MODELS) == 3


class TestTargetModules:
    """Tests for PEFT target module definitions."""

    def test_lora_modules_defined_for_all_architectures(self):
        for arch in MODELS:
            assert arch in LORA_TARGET_MODULES, f"Missing LoRA modules for {arch}"
            assert len(LORA_TARGET_MODULES[arch]) > 0

    def test_lora_ffn_modules_defined_for_all_architectures(self):
        for arch in MODELS:
            assert arch in LORA_FFN_TARGET_MODULES, f"Missing LoRA FFN modules for {arch}"
            assert len(LORA_FFN_TARGET_MODULES[arch]) > len(LORA_TARGET_MODULES[arch])

    def test_ia3_modules_defined_for_all_architectures(self):
        for arch in MODELS:
            assert arch in IA3_TARGET_MODULES, f"Missing IA3 modules for {arch}"
            assert arch in IA3_FF_MODULES, f"Missing IA3 FF modules for {arch}"

    def test_ia3_ff_modules_subset_of_target_modules(self):
        for arch in MODELS:
            for module in IA3_FF_MODULES[arch]:
                assert module in IA3_TARGET_MODULES[arch], (
                    f"IA3 FF module '{module}' not in target modules for {arch}"
                )


class TestDefaultConfigs:
    """Tests for default configuration values."""

    def test_training_defaults(self):
        assert DEFAULT_TRAINING["num_epochs"] == 2
        assert DEFAULT_TRAINING["per_device_batch_size"] == 8
        assert DEFAULT_TRAINING["gradient_accumulation_steps"] == 2
        assert DEFAULT_TRAINING["max_length"] == 256

    def test_training_seeds(self):
        assert 42 in DEFAULT_TRAINING["seeds"]
        assert 43 in DEFAULT_TRAINING["seeds"]
        assert len(DEFAULT_TRAINING["seeds"]) == 2

    def test_training_learning_rates(self):
        assert 1e-5 in DEFAULT_TRAINING["learning_rates"]
        assert 2e-5 in DEFAULT_TRAINING["learning_rates"]

    def test_lora_defaults(self):
        assert DEFAULT_LORA["r"] == 8
        assert DEFAULT_LORA["lora_alpha"] == 16
        assert DEFAULT_LORA["lora_dropout"] == 0.1
        assert DEFAULT_LORA["task_type"] == "SEQ_CLS"

    def test_rank_sensitivity_values(self):
        assert 8 in LORA_RANK_SENSITIVITY
        assert 16 in LORA_RANK_SENSITIVITY


class TestGetLoraConfig:
    """Tests for the get_lora_config function."""

    def test_returns_dict(self):
        config = get_lora_config("distilbert")
        assert isinstance(config, dict)

    def test_default_rank(self):
        config = get_lora_config("distilbert")
        assert config["r"] == 8

    def test_custom_rank(self):
        config = get_lora_config("distilbert", rank=16)
        assert config["r"] == 16

    def test_attention_only_modules(self):
        config = get_lora_config("distilbert", include_ffn=False)
        assert config["target_modules"] == LORA_TARGET_MODULES["distilbert"]

    def test_ffn_modules(self):
        config = get_lora_config("distilbert", include_ffn=True)
        assert config["target_modules"] == LORA_FFN_TARGET_MODULES["distilbert"]

    def test_all_architectures(self):
        for arch in MODELS:
            config = get_lora_config(arch)
            assert "r" in config
            assert "lora_alpha" in config
            assert "target_modules" in config
            assert "task_type" in config

    def test_invalid_architecture_raises(self):
        try:
            get_lora_config("invalid_model")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "invalid_model" in str(e)


class TestGetIA3Config:
    """Tests for the get_ia3_config function."""

    def test_returns_dict(self):
        config = get_ia3_config("distilbert")
        assert isinstance(config, dict)

    def test_has_required_keys(self):
        config = get_ia3_config("distilbert")
        assert "target_modules" in config
        assert "feedforward_modules" in config
        assert "task_type" in config

    def test_all_architectures(self):
        for arch in MODELS:
            config = get_ia3_config(arch)
            assert len(config["target_modules"]) > 0
            assert len(config["feedforward_modules"]) > 0

    def test_invalid_architecture_raises(self):
        try:
            get_ia3_config("invalid_model")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "invalid_model" in str(e)
