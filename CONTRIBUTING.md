# Contributing to torch-aac

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/verda-ai/torch-aac.git
cd torch-aac

# Create environment (uv recommended, pip works too)
uv venv && uv pip install -e ".[dev]"
# or: python -m venv .venv && pip install -e ".[dev]"

# Run tests
pytest

# Lint + type check
ruff check .
ruff format --check .
mypy torch_aac/
```

## Code Style

- Python 3.10+
- Type annotations on all functions (public and internal)
- Google-style docstrings on all public classes/methods
- `ruff` for linting and formatting
- `mypy --strict` clean (goal — some exceptions currently allowed)

## Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_filterbank.py

# With verbose output
pytest -v --tb=long
```

Tests run on CPU by default. GPU-specific tests are skipped when CUDA is unavailable.

## Architecture

```
torch_aac/
├── gpu/          # GPU-accelerated stages (PyTorch)
├── cpu/          # CPU stages (Huffman packing, bitstream)
├── tables/       # AAC spec tables (Huffman codebooks, SFB bands)
├── utils/        # Device management, audio I/O
├── encoder.py    # Encode mode orchestrator
├── differentiable.py  # Differentiable mode
└── config.py     # Configuration
```

See [`docs/technical.md`](docs/technical.md) for the full technical deep dive — optimization stories, bugs we found, and architectural decisions.

## Pull Requests

1. Fork and create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass: `pytest`
4. Ensure lint passes: `ruff check .`
5. Submit a PR with a clear description

## License

By contributing, you agree that your contributions will be licensed under the [Elastic License 2.0](LICENSE).
