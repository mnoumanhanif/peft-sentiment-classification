# Contributing to PEFT Sentiment Classification

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

1. Check [existing issues](https://github.com/mnoumanhanif/peft-sentiment-classification/issues) to avoid duplicates.
2. Use the **Bug Report** issue template.
3. Include your environment details (Python version, GPU, OS).
4. Provide steps to reproduce the issue.

### Suggesting Features

1. Open an issue using the **Feature Request** template.
2. Describe the feature and its use case.
3. Explain how it relates to the project goals.

### Submitting Code Changes

1. **Fork** the repository.
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the code style guidelines below.
4. **Write tests** for new functionality.
5. **Run tests** to ensure nothing is broken:
   ```bash
   pytest tests/
   ```
6. **Commit** with a clear message:
   ```bash
   git commit -m "Add: brief description of change"
   ```
7. **Push** to your fork and open a **Pull Request**.

## Code Style Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
- Use descriptive variable and function names.
- Add docstrings to all public functions and classes.
- Keep functions focused and single-purpose.
- Use type hints where appropriate.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/mnoumanhanif/peft-sentiment-classification.git
cd peft-sentiment-classification

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

## Commit Message Convention

Use clear, descriptive commit messages:

- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for updates to existing functionality
- `Docs:` for documentation changes
- `Refactor:` for code refactoring
- `Test:` for adding or updating tests

## Questions?

If you have questions, feel free to open an issue or reach out to the maintainers.

Thank you for contributing!
