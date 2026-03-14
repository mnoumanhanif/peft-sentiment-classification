# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] - 2026-03-14

### Added
- Modular Python source code in `src/` extracted from notebook
- Project documentation in `docs/` (setup, architecture, development guides)
- GitHub Actions CI workflow for linting and testing
- Issue templates (bug report, feature request) and PR template
- `CONTRIBUTING.md` with contribution guidelines
- `CHANGELOG.md` for tracking changes
- `requirements.txt` with pinned minimum dependency versions
- `.gitignore` for Python/Jupyter projects
- Unit tests in `tests/` for configuration and utility modules
- Experiment configuration files in `configs/`

### Changed
- Reorganized repository into a clean project structure
- Moved notebook to `notebooks/` directory
- Moved research paper PDF to `docs/` directory
- Improved README.md with updated structure, badges, and clearer instructions

## [1.0.0] - 2025-01-01

### Added
- Initial release with Jupyter notebook implementing 42-run experimental matrix
- Full Fine-Tuning, LoRA, QLoRA, and IA3 experiments
- Support for DistilBERT, RoBERTa-Base, and DeBERTa-v3 architectures
- IMDb sentiment classification on 50,000 samples
- Research paper in IEEE format
- MIT License
