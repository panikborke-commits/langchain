"""CLI interface for OpenClaw."""

import argparse
import sys


def main(args: list[str] | None = None) -> int:
    """Main entry point for the OpenClaw CLI.

    Args:
        args: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        prog="openclaw",
        description="OpenClaw - Global tool for LangChain",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.0.1",
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Command to execute",
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 0

    print(f"OpenClaw command: {parsed_args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
