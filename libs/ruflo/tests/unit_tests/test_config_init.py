"""Tests for `ruflo config init` command."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from langchain_ruflo.cli import (
    _slugify,
    _write_file,
    config_init,
    main,
)


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------


def test_slugify_replaces_spaces() -> None:
    assert _slugify("my project") == "my_project"


def test_slugify_replaces_hyphens() -> None:
    assert _slugify("my-project") == "my_project"


def test_slugify_lowercases() -> None:
    assert _slugify("MyProject") == "myproject"


def test_slugify_mixed() -> None:
    assert _slugify("My-Cool Project") == "my_cool_project"


# ---------------------------------------------------------------------------
# _write_file
# ---------------------------------------------------------------------------


def test_write_file_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "file.txt"
    result = _write_file(target, "hello")
    assert result is True
    assert target.read_text() == "hello"


def test_write_file_skips_existing_without_force(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("original")
    result = _write_file(target, "new content")
    assert result is False
    assert target.read_text() == "original"


def test_write_file_overwrites_with_force(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("original")
    result = _write_file(target, "new content", overwrite=True)
    assert result is True
    assert target.read_text() == "new content"


# ---------------------------------------------------------------------------
# config_init – non-interactive (name + description provided via kwargs)
# ---------------------------------------------------------------------------


def test_config_init_creates_expected_files(tmp_path: Path) -> None:
    config_init(tmp_path, name="acme", description="Test project")

    assert (tmp_path / "pyproject.toml").exists()
    assert (tmp_path / ".env.example").exists()
    assert (tmp_path / ".gitignore").exists()
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "src" / "acme" / "__init__.py").exists()
    assert (tmp_path / "src" / "acme" / "main.py").exists()


def test_config_init_pyproject_contains_name(tmp_path: Path) -> None:
    config_init(tmp_path, name="myapp", description="")
    content = (tmp_path / "pyproject.toml").read_text()
    assert 'name = "myapp"' in content


def test_config_init_pyproject_contains_langchain_dep(tmp_path: Path) -> None:
    config_init(tmp_path, name="myapp", description="")
    content = (tmp_path / "pyproject.toml").read_text()
    assert "langchain" in content


def test_config_init_env_example_contains_openai_key(tmp_path: Path) -> None:
    config_init(tmp_path, name="myapp", description="")
    content = (tmp_path / ".env.example").read_text()
    assert "OPENAI_API_KEY" in content


def test_config_init_env_example_contains_anthropic_key(tmp_path: Path) -> None:
    config_init(tmp_path, name="myapp", description="")
    content = (tmp_path / ".env.example").read_text()
    assert "ANTHROPIC_API_KEY" in content


def test_config_init_readme_contains_name(tmp_path: Path) -> None:
    config_init(tmp_path, name="myapp", description="A demo app")
    content = (tmp_path / "README.md").read_text()
    assert "myapp" in content
    assert "A demo app" in content


def test_config_init_gitignore_excludes_dotenv(tmp_path: Path) -> None:
    config_init(tmp_path, name="myapp", description="")
    content = (tmp_path / ".gitignore").read_text()
    assert ".env" in content


def test_config_init_module_name_slugified(tmp_path: Path) -> None:
    config_init(tmp_path, name="My Cool App", description="")
    assert (tmp_path / "src" / "my_cool_app" / "main.py").exists()


def test_config_init_force_overwrites(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("old content")
    config_init(tmp_path, name="myapp", description="", force=True)
    content = (tmp_path / "pyproject.toml").read_text()
    assert "old content" not in content
    assert "myapp" in content


def test_config_init_skips_existing_without_force(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("preserved")
    config_init(tmp_path, name="myapp", description="", force=False)
    assert (tmp_path / "pyproject.toml").read_text() == "preserved"


def test_config_init_defaults_name_to_directory(tmp_path: Path) -> None:
    named_dir = tmp_path / "coolproject"
    named_dir.mkdir()
    # stdin is not a tty in test, so name defaults to directory name
    config_init(named_dir)
    content = (named_dir / "pyproject.toml").read_text()
    assert "coolproject" in content


def test_config_init_creates_directory_if_missing(tmp_path: Path) -> None:
    new_dir = tmp_path / "brand_new"
    assert not new_dir.exists()
    config_init(new_dir, name="brand_new", description="")
    assert new_dir.is_dir()
    assert (new_dir / "pyproject.toml").exists()


# ---------------------------------------------------------------------------
# main() – argument parsing
# ---------------------------------------------------------------------------


def test_main_config_init_calls_config_init(tmp_path: Path) -> None:
    with (
        patch("sys.argv", ["ruflo", "config", "init", str(tmp_path), "--name", "demo"]),
        patch("langchain_ruflo.cli.config_init") as mock_init,
    ):
        main()
    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args
    assert call_kwargs.kwargs.get("name") == "demo" or call_kwargs.args[1:] == ()


def test_main_exits_without_subcommand() -> None:
    with patch("sys.argv", ["ruflo"]), pytest.raises(SystemExit):
        main()


def test_main_config_exits_without_subcommand() -> None:
    with patch("sys.argv", ["ruflo", "config"]), pytest.raises(SystemExit):
        main()


def test_main_config_init_force_flag(tmp_path: Path) -> None:
    with patch(
        "sys.argv",
        ["ruflo", "config", "init", str(tmp_path), "--name", "x", "--force"],
    ):
        main()
    assert (tmp_path / "pyproject.toml").exists()
