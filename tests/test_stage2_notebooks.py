import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    ROOT / "notebooks" / "stage2" / "stage2_phase0.ipynb",
    ROOT / "notebooks" / "stage2" / "stage2_phase1.ipynb",
    ROOT / "notebooks" / "stage2" / "stage2_phase2a_smoke.ipynb",
    ROOT / "notebooks" / "stage2" / "stage2_phase2a_2.ipynb",
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


def test_r4_notebook_validates_resampled_speech_length():
    path = ROOT / "notebooks" / "stage2" / "stage2_phase2a_2.ipynb"
    notebook = json.loads(path.read_text())
    forward_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "FORWARD: PASS" in "".join(cell["source"])
    )
    assert "int(speech_mask[i].sum())" in forward_cell
    assert 'int(batch["speech_mask"][i].sum())' not in forward_cell
