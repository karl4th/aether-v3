"""Static-analysis checks (ruff, mypy) as pytest tests, so lint/type
regressions surface alongside functional ones instead of needing a separate
CI step remembered by hand.

Needs the `dev` dependency group: `uv sync --group dev`. Skips (rather than
failing) if a tool isn't installed, so the rest of the suite still runs
without it.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LINT_TARGETS = ["src", "tests", "scripts"]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)


def _skip_if_missing(result: subprocess.CompletedProcess, module: str) -> None:
    if f"No module named {module}" in result.stderr:
        pytest.skip(f"{module} not installed - run `uv sync --group dev`")


def test_ruff_check() -> None:
    result = _run([sys.executable, "-m", "ruff", "check", *LINT_TARGETS])
    _skip_if_missing(result, "ruff")
    assert result.returncode == 0, result.stdout + result.stderr


def test_ruff_format_check() -> None:
    result = _run([sys.executable, "-m", "ruff", "format", "--check", *LINT_TARGETS])
    _skip_if_missing(result, "ruff")
    assert result.returncode == 0, result.stdout + result.stderr


def test_mypy() -> None:
    result = _run([sys.executable, "-m", "mypy", "src"])
    _skip_if_missing(result, "mypy")
    assert result.returncode == 0, result.stdout + result.stderr
