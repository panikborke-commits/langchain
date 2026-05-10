# langchain-openclaw

This package provides a global CLI tool for OpenClaw integration with LangChain.

## Installation

```bash
pip install langchain-openclaw
# or
uv add langchain-openclaw
```

## Usage

```bash
openclaw --help
```

## Development

Install the package in development mode:

```bash
cd libs/openclaw
uv sync --all-groups
```

Run tests:

```bash
uv run --group test pytest
```

Run linting:

```bash
uv run --group lint ruff check .
uv run --group lint ruff format .
```

Run type checking:

```bash
uv run --group typing mypy .
```
