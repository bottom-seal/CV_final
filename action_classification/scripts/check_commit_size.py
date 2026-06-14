#!/usr/bin/env python3
"""Fail if files selected for a commit exceed a conservative size limit."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-mb", type=float, default=10.0)
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()

    command = ["git", "diff", "--cached", "--name-only", "-z"]
    if not args.staged:
        command = [
            "git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"
        ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        print("Run this command after initializing the project Git repository.")
        return 2

    limit = int(args.max_mb * 1024 * 1024)
    oversized = []
    for raw_name in result.stdout.split(b"\0"):
        if not raw_name:
            continue
        path = Path(raw_name.decode())
        if path.is_file() and path.stat().st_size > limit:
            oversized.append((path, path.stat().st_size))

    if not oversized:
        print(f"No selected files exceed {args.max_mb:g} MB.")
        return 0

    print("Refusing large files:")
    for path, size in sorted(oversized, key=lambda item: item[1], reverse=True):
        print(f"  {size / 1024 / 1024:.1f} MB  {path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
