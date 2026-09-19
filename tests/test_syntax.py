"""Every .py file under src/, scripts/, and tests/ must at least parse.

A formalized, per-file version of the `py_compile $(find ... -name '*.py')`
check used ad hoc during development - runs as an ordinary test now, with
one parametrized case per file instead of one opaque pass/fail for the
whole tree.
"""
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _all_python_files() -> list[Path]:
    files: list[Path] = []
    for d in ("src", "scripts", "tests"):
        files.extend((REPO_ROOT / d).rglob("*.py"))
    return sorted(files)


@pytest.mark.parametrize(
    "path", _all_python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_file_parses(path: Path) -> None:
    ast.parse(path.read_text(), filename=str(path))
