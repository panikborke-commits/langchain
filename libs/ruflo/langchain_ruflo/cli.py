"""CLI for scaffolding LangChain project configurations."""

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Template content
# ---------------------------------------------------------------------------

_PYPROJECT_TEMPLATE = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{name}"
version = "0.1.0"
description = "{description}"
requires-python = ">=3.10"
dependencies = [
    "langchain>=0.3",
    "langchain-core>=0.3",
]

[dependency-groups]
test = [
    "pytest>=9.0",
]
"""

_ENV_EXAMPLE_TEMPLATE = """\
# Copy this file to .env and fill in your API keys.
# Never commit .env to version control.

# OpenAI
OPENAI_API_KEY=

# Anthropic
ANTHROPIC_API_KEY=

# LangSmith (optional – for tracing)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT={name}
"""

_MAIN_TEMPLATE = """\
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_chain(model: object) -> object:
    \"\"\"Return a simple prompt | model | parser chain.

    Args:
        model: Any LangChain-compatible chat model.

    Returns:
        A runnable chain that accepts a `topic` input and returns a string.
    \"\"\"
    prompt = ChatPromptTemplate.from_template("Tell me a fun fact about {{topic}}.")
    return prompt | model | StrOutputParser()
"""

_GITIGNORE_TEMPLATE = """\
.env
__pycache__/
*.py[cod]
.mypy_cache/
.ruff_cache/
dist/
"""

_README_TEMPLATE = """\
# {name}

{description}

## Setup

```bash
cp .env.example .env
# Fill in your API keys in .env

uv sync --group test
```

## Run

```python
from {module}.main import build_chain

# from langchain_openai import ChatOpenAI
# chain = build_chain(ChatOpenAI(model="gpt-4.1"))
# print(chain.invoke({{"topic": "penguins"}}))
```
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert arbitrary text to a Python-safe module name.

    Args:
        text: Raw project name entered by the user.

    Returns:
        Lowercase, hyphen-free string with spaces replaced by underscores,
        suitable for use as a Python package/module identifier.
    """
    return text.lower().replace("-", "_").replace(" ", "_")


def _write_file(path: Path, content: str, *, overwrite: bool = False) -> bool:
    """Write `content` to `path`, creating parent directories as needed.

    Args:
        path: Destination file path.
        content: Text content to write.
        overwrite: When `False`, skip files that already exist.

    Returns:
        `True` if the file was written, `False` if it was skipped.
    """
    if path.exists() and not overwrite:
        print(f"  skip  {path}  (already exists)")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  create  {path}")
    return True


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def config_init(  # noqa: C901
    directory: Path,
    *,
    name: str | None = None,
    description: str | None = None,
    force: bool = False,
) -> None:
    """Scaffold a new LangChain project in `directory`.

    Generates the following files:

    ```
    <directory>/
    ├── pyproject.toml
    ├── .env.example
    ├── .gitignore
    ├── README.md
    └── src/<module>/
        ├── __init__.py
        └── main.py
    ```

    Args:
        directory: Root directory for the new project. Created if it does not
            exist.
        name: Project name. Defaults to the directory's base name when not
            provided interactively or via flag.
        description: Short project description. Defaults to an empty string.
        force: When `True`, overwrite files that already exist.
    """
    directory = directory.resolve()

    # Determine project name
    if name is None:
        default_name = directory.name
        if sys.stdin.isatty():
            raw = input(f"Project name [{default_name}]: ").strip()
            name = raw or default_name
        else:
            name = default_name

    # Determine project description
    if description is None:
        if sys.stdin.isatty():
            description = input("Short description []: ").strip()
        else:
            description = ""

    module = _slugify(name)

    print(f"\nScaffolding project '{name}' in {directory}\n")

    _write_file(
        directory / "pyproject.toml",
        _PYPROJECT_TEMPLATE.format(name=name, description=description),
        overwrite=force,
    )
    _write_file(
        directory / ".env.example",
        _ENV_EXAMPLE_TEMPLATE.format(name=name),
        overwrite=force,
    )
    _write_file(
        directory / ".gitignore",
        _GITIGNORE_TEMPLATE,
        overwrite=force,
    )
    _write_file(
        directory / "README.md",
        _README_TEMPLATE.format(name=name, description=description, module=module),
        overwrite=force,
    )
    _write_file(
        directory / "src" / module / "__init__.py",
        "",
        overwrite=force,
    )
    _write_file(
        directory / "src" / module / "main.py",
        _MAIN_TEMPLATE,
        overwrite=force,
    )

    print(f"\n✓ Project '{name}' initialized. Next steps:")
    print(f"  cd {directory}")
    print(f"  cp .env.example .env  # then fill in your API keys")
    print(f"  uv sync")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="ruflo – LangChain project scaffolding",
        prog="ruflo",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # config sub-command group
    config_parser = subparsers.add_parser("config", help="Manage project configuration")
    config_subparsers = config_parser.add_subparsers(
        dest="config_command", required=True
    )

    # config init
    init_parser = config_subparsers.add_parser(
        "init",
        help="Scaffold a new LangChain project",
    )
    init_parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        type=Path,
        help="Target directory for the new project (default: current directory)",
    )
    init_parser.add_argument(
        "--name",
        default=None,
        help="Project name (default: directory name)",
    )
    init_parser.add_argument(
        "--description",
        default=None,
        help="Short project description",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )

    args = parser.parse_args()

    if args.command == "config" and args.config_command == "init":
        config_init(
            args.directory,
            name=args.name,
            description=args.description,
            force=args.force,
        )


if __name__ == "__main__":
    main()
