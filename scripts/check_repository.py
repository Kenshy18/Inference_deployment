#!/usr/bin/env python3
"""Run the repository's deterministic, non-GPU quality checks.

This script is the canonical developer entry point because production WSL
images do not necessarily include ``make``.  It intentionally excludes model
inference and long media tests; those belong to deployment acceptance tests.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = (
    "inference",
    "postprocess",
    "orchestration",
    "overlay",
    "deployment",
    "gui",
)


def _run(command: Sequence[str], *, cwd: Path, pythonpath: str | None = None) -> None:
    environment = os.environ.copy()
    if pythonpath is not None:
        previous = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            pythonpath if not previous else os.pathsep.join((pythonpath, previous))
        )
    rendered = " ".join(command)
    print(f"\n[{cwd.relative_to(ROOT) or Path('.')}] $ {rendered}", flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def _check_component(name: str) -> None:
    python = sys.executable
    if name == "inference":
        _run(
            (python, "-m", "pytest", "tests", "-q"),
            cwd=ROOT / "InstanceSegmentation" / "inference",
            pythonpath=os.pathsep.join(
                (str(ROOT / "InstanceSegmentation" / "inference"), str(ROOT))
            ),
        )
    elif name == "postprocess":
        _run(
            (python, "-m", "pytest", "tests", "-q"),
            cwd=ROOT / "postprocess",
            pythonpath=str(ROOT / "postprocess"),
        )
    elif name == "orchestration":
        _run(
            (python, "-m", "unittest", "discover", "-s", "orchestration/tests", "-v"),
            cwd=ROOT,
            pythonpath=str(ROOT),
        )
    elif name == "overlay":
        _run(
            (python, "-m", "unittest", "discover", "-s", "tests", "-v"),
            cwd=ROOT / "overlay",
            pythonpath=str(ROOT / "overlay" / "src"),
        )
    elif name == "deployment":
        _run(
            (
                python,
                "-m",
                "unittest",
                "discover",
                "-s",
                "deployment_tests/scripts",
                "-p",
                "test_*.py",
                "-v",
            ),
            cwd=ROOT,
            pythonpath=str(ROOT),
        )
    elif name == "gui":
        _run(("npm", "run", "typecheck"), cwd=ROOT / "gui")
        _run(("npm", "test"), cwd=ROOT / "gui")
    else:  # pragma: no cover - argparse constrains this value.
        raise ValueError(f"unknown component: {name}")


def _compile_python() -> None:
    targets = (
        "InstanceSegmentation/inference",
        "orchestration",
        "postprocess",
        "overlay/src",
        "deployment",
        "deployment_tests",
    )
    excluded = r"(^|/)(\.runtime|\.venv|node_modules|output|build|dist|__pycache__)(/|$)"
    _run(
        (sys.executable, "-m", "compileall", "-q", "-x", excluded, *targets),
        cwd=ROOT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "components",
        nargs="*",
        metavar="COMPONENT",
        help="components to check (default: all)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip Python bytecode compilation",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="compile Python sources without running component tests",
    )
    args = parser.parse_args()

    unknown = sorted(set(args.components) - set(COMPONENTS))
    if unknown:
        parser.error(
            f"unknown component(s): {', '.join(unknown)}; "
            f"choose from {', '.join(COMPONENTS)}"
        )
    if args.no_compile and args.compile_only:
        parser.error("--no-compile and --compile-only cannot be combined")
    if not args.no_compile:
        _compile_python()
    if args.compile_only:
        print("\nPython compilation passed.")
        return 0
    for component in args.components or COMPONENTS:
        _check_component(component)
    print("\nRepository checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
