import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    ROOT / "notebooks" / "stage2" / "stage2_phase0.ipynb",
    ROOT / "notebooks" / "stage2" / "stage2_phase1.ipynb",
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
        setup = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "logging.basicConfig" in "".join(cell["source"])
        )
        tree = ast.parse(setup)
        import_line = min(
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Import) and any(alias.name == "logging" for alias in node.names)
        )
        basic_config_line = setup[: setup.index("logging.basicConfig")].count("\n") + 1
        assert import_line < basic_config_line
