"""Tests for the OpenClaw CLI."""

import pytest

from langchain_openclaw.cli import main


def test_main_no_args() -> None:
    """Test main function with no arguments."""
    result = main([])
    assert result == 0


def test_main_version() -> None:
    """Test main function with --version flag."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0


def test_main_with_command() -> None:
    """Test main function with a command."""
    result = main(["test-command"])
    assert result == 0
