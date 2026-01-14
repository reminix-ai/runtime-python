#!/usr/bin/env python3
"""Run all code quality checks."""

import subprocess
import sys


def main() -> int:
    """Run format check, lint, and typecheck."""
    commands = [
        ["ruff", "format", "--check", "."],
        ["ruff", "check", "."],
        ["pyright"],
    ]

    for cmd in commands:
        print(f"\n> {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return result.returncode

    print("\n✓ All checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
