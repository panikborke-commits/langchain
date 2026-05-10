"""Tests for the OpenClaw CLI."""

from langchain_openclaw.cli import main


def test_main_no_args() -> None:
    """Test main function with no arguments."""
    result = main([])
    assert result == 0


def test_main_version() -> None:
    """Test main function with --version flag."""
    # This will exit with SystemExit, so we expect an exception
    try:
        main(["--version"])
        # If we get here, the version flag didn't work as expected
        assert False, "Expected SystemExit for --version"
    except SystemExit as e:
        assert e.code == 0


def test_main_with_command() -> None:
    """Test main function with a command."""
    result = main(["test-command"])
    assert result == 0
