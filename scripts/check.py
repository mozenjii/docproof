"""Run every verification gate in order, cheapest first, and stop at the first failure.

This is the one command behind local development and CI, so the two cannot drift: CI calls
exactly this script. Run it with ``uv run python scripts/check.py``.

Deliberate boundary: the script only orchestrates. Each gate's rules live in the tool's own
configuration in ``pyproject.toml``; nothing here relaxes or overrides them.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Gate:
    """One named verification step and the command that runs it."""

    name: str
    command: tuple[str, ...]
    needs_network: bool = False


def _uv_executable() -> str:
    """Return the uv executable, preferring the one that launched this process.

    Raises:
        SystemExit: if uv cannot be found, because the lockfile gate cannot run without it and a
            gate that silently does not run is worse than a failed one.
    """
    found = os.environ.get("UV") or shutil.which("uv")
    if not found:
        raise SystemExit(
            "check: uv not found; run this script with `uv run python scripts/check.py`"
        )
    return found


def gates() -> list[Gate]:
    """Return the gates in the order they run, cheapest and most likely to fail first."""
    py = sys.executable
    return [
        Gate("lockfile", (_uv_executable(), "lock", "--check")),
        Gate("provenance", (py, "-m", "docproof.provenance")),
        Gate("format", (py, "-m", "ruff", "format", "--check", ".")),
        Gate("lint", (py, "-m", "ruff", "check", ".")),
        Gate("types", (py, "-m", "pyright")),
        Gate("tests", (py, "-m", "pytest", "--cov", "--cov-report=term")),
        Gate("dependency audit", (py, "-m", "pip_audit", "--skip-editable"), needs_network=True),
    ]


def run(selected: list[Gate]) -> int:
    """Run the gates in order and return a process exit code.

    Args:
        selected: the gates to run, already in execution order.

    Returns:
        0 if every gate passed, otherwise the exit code of the first gate that failed.
    """
    for gate in selected:
        print(f"\n== {gate.name}: {' '.join(gate.command[1:])}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(gate.command, check=False)
        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            print(f"\nFAILED: {gate.name} (exit {result.returncode}, {elapsed:.1f}s)", flush=True)
            return result.returncode
        print(f"-- {gate.name} passed in {elapsed:.1f}s", flush=True)
    print(f"\nAll {len(selected)} gates passed.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the selected gates."""
    parser = argparse.ArgumentParser(description="Run every verification gate in order.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip gates that need the network (the dependency audit); the skip is printed",
    )
    args = parser.parse_args(argv)
    selected = gates()
    if args.offline:
        skipped = [g.name for g in selected if g.needs_network]
        selected = [g for g in selected if not g.needs_network]
        print(f"check: offline, NOT running: {', '.join(skipped)}", flush=True)
    return run(selected)


if __name__ == "__main__":
    sys.exit(main())
