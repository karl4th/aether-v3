import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    ROOT / "notebooks" / "stage2_smoke_test.ipynb",
    ROOT / "notebooks" / "stage2_train.ipynb",
]


def test_stage2_notebook_cells_compile_and_contain_no_literal_hf_token():
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"{path.name}:cell{index}", "exec")
            assert 'os.environ["HF_TOKEN"] = "hf_' not in source


def test_setup_imports_logging_before_using_it():
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text())
        setup = "".join(notebook["cells"][1]["source"])
        assert setup.index("import os, subprocess, sys, logging") < setup.index(
            "logging.basicConfig"
        )
