from aether_v3.training.artifacts import _files_for_backup


def test_backup_manifest_excludes_predictions_and_temporary_files(tmp_path):
    (tmp_path / "selections").mkdir()
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "predictions").mkdir()
    (tmp_path / "config.resolved.yaml").write_text("train: {}")
    (tmp_path / "selections" / "best_wer.json").write_text("{}")
    (tmp_path / "checkpoints" / "last.pt").write_bytes(b"checkpoint")
    (tmp_path / "checkpoints" / "last.pt.tmp").write_bytes(b"partial")
    (tmp_path / "predictions" / "large.jsonl").write_text("ignored")

    relative = {str(path.relative_to(tmp_path)) for path in _files_for_backup(tmp_path)}

    assert relative == {
        "config.resolved.yaml",
        "selections/best_wer.json",
        "checkpoints/last.pt",
    }
