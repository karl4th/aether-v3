import json

import pytest

from aether_v3.config import ExperimentConfig
from aether_v3.training.run import MetricSelections, RunDirectory


def test_each_new_run_gets_an_exclusive_directory(tmp_path):
    config = ExperimentConfig()
    config.train.runs_dir = str(tmp_path)
    config.train.run_id = "fixed-run"
    run = RunDirectory.create(config)

    assert run.path == tmp_path / "fixed-run"
    assert (run.path / "config.resolved.yaml").exists()
    assert (run.path / "provenance.json").exists()
    assert json.loads((run.path / "status.json").read_text())["state"] == "created"
    with pytest.raises(FileExistsError):
        RunDirectory.create(config)


def test_metric_selections_track_categories_independently(tmp_path):
    config = ExperimentConfig()
    config.train.runs_dir = str(tmp_path)
    config.train.run_id = "selection-run"
    run = RunDirectory.create(config)
    checkpoint = run.path / "checkpoints" / "step_10_model.pt"
    checkpoint.touch()
    selections = MetricSelections(run)

    improved = selections.update(
        {"eval_loss": 2.0, "eval_wer": 0.3, "eval_cer": 0.1}, 10, checkpoint
    )
    assert set(improved) == {"eval_loss", "eval_wer", "eval_cer"}
    assert (run.path / "selections" / "best_wer.json").exists()

    improved = selections.update(
        {"eval_loss": 2.1, "eval_wer": 0.25, "eval_cer": 0.11}, 20, checkpoint
    )
    assert improved == ["eval_wer"]


def test_future_dataset_aware_metrics_are_supported(tmp_path):
    config = ExperimentConfig()
    config.train.runs_dir = str(tmp_path)
    config.train.run_id = "slice-selection-run"
    run = RunDirectory.create(config)
    checkpoint = run.path / "checkpoints" / "step_10_model.pt"
    checkpoint.touch()
    selections = MetricSelections(run)

    improved = selections.update(
        {
            "macro_domain_wer": 0.2,
            "short_query_wer": 0.15,
            "streaming_wer": 0.22,
            "catastrophic_failure_rate": 0.01,
        },
        10,
        checkpoint,
    )

    assert set(improved) == {
        "macro_domain_wer",
        "short_query_wer",
        "streaming_wer",
        "catastrophic_failure_rate",
    }


def test_metric_selections_ignore_nonfinite_values(tmp_path):
    config = ExperimentConfig()
    config.train.runs_dir = str(tmp_path)
    config.train.run_id = "nonfinite-selection-run"
    run = RunDirectory.create(config)
    selections = MetricSelections(run)

    assert selections.improvements({"short_query_wer": float("nan")}) == []
    assert selections.improvements({"eval_loss": float("inf")}) == []


def test_resume_opens_existing_run_without_overwriting_provenance(tmp_path):
    config = ExperimentConfig()
    config.train.runs_dir = str(tmp_path)
    config.train.run_id = "resume-run"
    original = RunDirectory.create(config)
    provenance_before = (original.path / "provenance.json").read_text()

    resumed = RunDirectory.resume(original.path)

    assert resumed.run_id == "resume-run"
    assert (original.path / "provenance.json").read_text() == provenance_before
