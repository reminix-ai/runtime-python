#!/usr/bin/env python3
import os
from pathlib import Path

import tomllib


def main() -> int:
    tag = os.environ.get("GITHUB_REF_NAME", "").removeprefix("v")
    if not tag:
        print("GITHUB_REF_NAME is not set.")
        return 1

    packages_dir = Path("packages")
    mismatches = []
    for pyproject in packages_dir.glob("*/pyproject.toml"):
        data = tomllib.loads(pyproject.read_text())
        version = data.get("project", {}).get("version")
        if version != tag:
            mismatches.append(f"{pyproject.parent.name}: {version}")

    if mismatches:
        print(f"Tag v{tag} does not match package versions:")
        for item in mismatches:
            print(f"- {item}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
