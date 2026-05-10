.PHONY: test lint format type-check all

test:
	uv run --group test pytest

lint:
	uv run --group lint ruff check .
	uv run --group typing mypy .

format:
	uv run --group lint ruff format .

type-check:
	uv run --group typing mypy .

all: lint test
