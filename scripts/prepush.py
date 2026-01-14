#!/usr/bin/env python3
"""Run all checks and tests before pushing."""

import subprocess
import sys


def main() -> int:
    """Run format check, lint, typecheck, and tests."""
    commands = [
        ["ruff", "format", "--check", "."],
        ["ruff", "check", "."],
        ["pyright"],
        ["pytest"],
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
