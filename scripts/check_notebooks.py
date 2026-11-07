#!/usr/bin/env python
"""Validate notebooks and optionally execute temporary copies with this interpreter.

Usage: python scripts/check_notebooks.py [--execute] [--timeout 180]
Execution uses temporary copies and leaves saved notebooks unchanged. Kernels need
local loopback sockets, as standard Jupyter kernels do. No external network is used.
Temporary copies disable optional interactive viewer launches.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--paper", action="store_true", help="Check the eight paper notebooks")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = sorted(
        p
        for p in (root / "notebooks").rglob("*.ipynb")
        if "analysis_outputs" not in p.relative_to(root / "notebooks").parts
        and ".ipynb_checkpoints" not in p.parts
    )
    if args.paper:
        paper_names = {
            "dfst_arithmetic.ipynb",
            "dfst_softmax_arithmetic.ipynb",
            "dfst_universal.ipynb",
            "dfst_softmax_universal.ipynb",
            "front_page_summary.ipynb",
            "metric_FST_extraction.ipynb",
            "anneal_dfst_softmax_to_fst.ipynb",
            "training_dynamics.ipynb",
        }
        paths = [p for p in paths if p.name in paper_names]
        if {p.name for p in paths} != paper_names:
            raise SystemExit("A paper notebook is missing")
    if not paths:
        raise SystemExit("No notebooks found")
    with tempfile.TemporaryDirectory(prefix="del-notebooks-") as folder:
        scratch = Path(folder)
        if args.execute:
            from jupyter_client import KernelManager
            from jupyter_client.kernelspec import KernelSpecManager
            from nbclient import NotebookClient

            spec = scratch / "kernels" / "validation"
            spec.mkdir(parents=True)
            (spec / "kernel.json").write_text(
                json.dumps(
                    {
                        "argv": [
                            sys.executable,
                            "-m",
                            "ipykernel_launcher",
                            "-f",
                            "{connection_file}",
                        ],
                        "display_name": "Current Python validation",
                        "language": "python",
                    }
                )
            )
            manager = KernelSpecManager(kernel_dirs=[str(spec.parent)])
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            source = "\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
            compile(source, str(path), "exec")
            lint = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    "--select",
                    "F",
                    "--stdin-filename",
                    path.stem + ".py",
                    "-",
                ],
                input=source,
                text=True,
                capture_output=True,
            )
            if lint.returncode:
                raise SystemExit(f"{path}:\n{lint.stdout}{lint.stderr}")
            if args.execute:
                work = scratch / path.stem
                work.mkdir()
                temporary = copy.deepcopy(notebook)
                for cell in temporary.cells:
                    if cell.cell_type == "code":
                        cell.source = re.sub(
                            r"(?m)^(LAUNCH_(?:LEARNED_)?ANIMATION)\s*=\s*True\s*$",
                            r"\1 = False",
                            cell.source,
                        )
                temporary.metadata.kernelspec = {
                    "name": "validation",
                    "display_name": "Current Python validation",
                    "language": "python",
                }
                nbformat.write(temporary, work / path.name)
                env = dict(os.environ)
                env.update(
                    {
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "MPLBACKEND": "Agg",
                        "MPLCONFIGDIR": str(scratch / "matplotlib"),
                        "IPYTHONDIR": str(scratch / "ipython"),
                        "DEL_ANALYSIS_DIR": str(work / "outputs"),
                        "DEL_REPO_ROOT": str(root),
                        "PYTHONPATH": str(root / "src") + os.pathsep + env.get("PYTHONPATH", ""),
                    }
                )
                # Use explicit notebook selections rather than a shell-level run override.
                env.pop("DEL_RUN_DIR", None)
                km = KernelManager(
                    kernel_name="validation",
                    kernel_spec_manager=manager,
                    connection_file=str(work / "kernel.json"),
                )
                client = NotebookClient(
                    temporary,
                    km=km,
                    timeout=args.timeout,
                    resources={"metadata": {"path": str(work)}},
                )
                try:
                    client.execute(env=env)
                finally:
                    if km.has_kernel:
                        km.shutdown_kernel(now=True)
            print(
                f"{'EXECUTED' if args.execute else 'VALIDATED'} {path.relative_to(root)}",
                flush=True,
            )
    print(f"{len(paths)} notebooks passed.", flush=True)


if __name__ == "__main__":
    main()
