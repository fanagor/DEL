#!/usr/bin/env python
"""Refresh the selected final checkpoints, configurations, and evaluations."""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from differentiable_exact_learning.analysis.release import export_run, read_catalog  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Private export manifest outside the public repository",
    )
    args = parser.parse_args()
    rows = []
    for entry in read_catalog(args.root, args.manifest):
        row = export_run(args.root, entry)
        rows.append(row)
        print(f"{row['status']:20s} {row['path']}", flush=True)
    print(dict(Counter(row["status"] for row in rows)))


if __name__ == "__main__":
    main()
