#!/usr/bin/env python3
"""Bump version numbers across all packages in the Python monorepo.

Usage:
    # Dry run to see what would change
    uv run bump-version patch --dry-run
    # or: python scripts/bump_version.py patch --dry-run
    
    # Bump patch version (0.0.0 -> 0.0.1)
    uv run bump-version patch
    
    # Bump minor version (0.0.0 -> 0.1.0)
    uv run bump-version minor
    
    # Bump major version (0.0.0 -> 1.0.0)
    uv run bump-version major
    
The script will:
- Update all package versions in packages/ directory
- Update cross-package dependencies to use the new version
- Keep workspace references unchanged (they resolve automatically)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Literal


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse a version string into (major, minor, patch)."""
    parts = version.split(".")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def format_version(major: int, minor: int, patch: int) -> str:
    """Format version tuple into string."""
    return f"{major}.{minor}.{patch}"


def bump_version(version: str, bump_type: Literal["major", "minor", "patch"]) -> str:
    """Bump a version string."""
    major, minor, patch = parse_version(version)
    
    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")
    
    return format_version(major, minor, patch)


def get_package_name_from_pyproject(content: str) -> str | None:
    """Extract package name from pyproject.toml content."""
    match = re.search(r'^name\s*=\s*"([^"]+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def update_pyproject_toml(file_path: Path, old_version: str, new_version: str, root: Path, dry_run: bool = False) -> bool:
    """Update version in pyproject.toml file."""
    content = file_path.read_text()
    original_content = content
    package_name = get_package_name_from_pyproject(content)
    
    # For managed packages (in packages/), always update version to new_version to keep them in sync
    if is_managed_package(file_path, root):
        # Update version in [project] section
        pattern = r'(^version\s*=\s*")([^"]+)(")'
        def replace_version(match):
            return f'{match.group(1)}{new_version}{match.group(3)}'
        content = re.sub(pattern, replace_version, content, flags=re.MULTILINE)
    
    # Update dependencies that reference packages in packages/ directory
    # Match any package name that looks like it's from our monorepo (reminix-*)
    # Pattern: "package-name==0.0.0" or "package-name>=0.0.0"
    pattern = rf'("(reminix-[a-z-]+))(==|>=)({re.escape(old_version)})"'
    def replace_dep(match):
        pkg_name = match.group(2)
        return f'"{pkg_name}=={new_version}"'
    content = re.sub(pattern, replace_dep, content)
    
    if content != original_content:
        if not dry_run:
            file_path.write_text(content)
        return True
    return False


def is_managed_package(file_path: Path, root: Path) -> bool:
    """Check if this is a package we manage (in packages/, not root or examples/)."""
    rel_path = file_path.relative_to(root)
    # Only files in packages/ (not root, not examples/)
    return "packages" in rel_path.parts and "examples" not in rel_path.parts




def find_package_files(root: Path) -> list[tuple[Path, str]]:
    """Find all pyproject.toml files in packages/ and their current versions."""
    files = []
    
    # Find Python packages in packages/ directory
    packages_dir = root / "packages"
    if not packages_dir.exists():
        return files
    
    for pyproject in packages_dir.rglob("pyproject.toml"):
        # Skip if in ignored directories
        if any(part.startswith(".") for part in pyproject.parts):
            continue
        
        content = pyproject.read_text()
        # Extract version from [project] section
        match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        if match:
            files.append((pyproject, match.group(1)))
    
    return files


def get_current_version(root: Path) -> str | None:
    """Get the current version from the first package found."""
    files = find_package_files(root)
    if files:
        return files[0][1]
    return None


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Bump version numbers across all packages")
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch"],
        help="Type of version bump"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Root directory for Python workspace (default: parent of scripts/)"
    )
    
    args = parser.parse_args()
    
    python_root = Path(args.root)
    
    # Get current version
    current_version = get_current_version(python_root)
    if not current_version:
        print("Error: Could not determine current version")
        return 1
    
    new_version = bump_version(current_version, args.bump_type)
    
    print(f"Bumping version from {current_version} to {new_version} ({args.bump_type})")
    if args.dry_run:
        print("\n[DRY RUN] Would update the following files:")
    else:
        print("\nUpdating files...")
    
    updated_count = 0
    
    # Update Python packages (only in packages/, not root or examples/)
    for pyproject in python_root.rglob("pyproject.toml"):
        if any(part.startswith(".") for part in pyproject.parts):
            continue
        # Skip root and examples
        rel_path = pyproject.relative_to(python_root)
        if len(rel_path.parts) == 1 or "examples" in rel_path.parts:
            continue
        
        if update_pyproject_toml(pyproject, current_version, new_version, python_root, dry_run=args.dry_run):
            print(f"  ✓ {pyproject.relative_to(python_root)}")
            updated_count += 1
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would update {updated_count} files")
        print("Run without --dry-run to apply changes")
    else:
        print(f"\n✓ Updated {updated_count} files")
        print(f"\nNew version: {new_version}")
        print("\nNext steps:")
        print("  1. Review the changes")
        print("  2. Commit the version bump")
        print("  3. Tag the release: git tag v{new_version}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
