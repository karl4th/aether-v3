import json
import sys
from types import SimpleNamespace

import pytest

from aether_v3.config import ExperimentConfig
from aether_v3.training.monitoring import start_wandb_monitor


class FakeRun:
    url = "https://wandb.example/runs/stage1-test"
    project_url = "https://wandb.example/projects/aether"
    entity = "manifestro"

    def __init__(self):
        self.metrics = []
        self.logs = []
        self.finished_with = None
        self.fail_logging = False

    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def log(self, payload, **kwargs):
        if self.fail_logging:
            raise ConnectionError("offline")
        self.logs.append((payload, kwargs))

    def finish(self, *, exit_code):
        self.finished_with = exit_code


class FakeTable:
    def __init__(self, *, columns, data):
        self.columns = columns
        self.data = data


def _config(*, required=True):
    config = ExperimentConfig()
    config.train.wandb_project = "manifestro-aether-stage1"
    config.train.wandb_required = required
    return config


def _install_fake_wandb(monkeypatch, run):
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Table=FakeTable))
    return calls


def test_monitoring_is_disabled_when_no_project(tmp_path):
    assert start_wandb_monitor(ExperimentConfig(), tmp_path, "run-1") is None


def test_required_monitoring_rejects_missing_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        start_wandb_monitor(_config(), tmp_path, "run-1")


def test_monitor_uses_stable_run_id_and_writes_public_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret-value")
    run = FakeRun()
    calls = _install_fake_wandb(monkeypatch, run)

    monitor = start_wandb_monitor(_config(), tmp_path, "stage1-test")

    assert monitor is not None
    assert calls[0]["id"] == "stage1-test"
    assert calls[0]["name"] == "stage1-test"
    assert calls[0]["resume"] == "allow"
    assert calls[0]["dir"] == str(tmp_path)
    metadata_text = (tmp_path / "monitoring.json").read_text()
    metadata = json.loads(metadata_text)
    assert metadata["run_url"] == run.url
    assert metadata["project_url"] == run.project_url
    assert "secret-value" not in metadata_text


def test_monitor_can_use_local_spool_for_network_volume(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    monkeypatch.setenv("AETHER_WANDB_DIR", "/tmp/aether-wandb")
    run = FakeRun()
    calls = _install_fake_wandb(monkeypatch, run)

    assert start_wandb_monitor(_config(), tmp_path, "run-1") is not None
    assert calls[0]["dir"] == "/tmp/aether-wandb"


def test_logging_failure_disables_remote_logging_without_raising(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    run = FakeRun()
    _install_fake_wandb(monkeypatch, run)
    monitor = start_wandb_monitor(_config(), tmp_path, "run-1")
    assert monitor is not None
    run.fail_logging = True

    error = monitor.log({"train/loss": 1.0}, step=1)
    second_error = monitor.log({"train/loss": 0.5}, step=2)

    assert error == "ConnectionError: offline"
    assert second_error is None
    assert monitor.failed is True


def test_evaluation_logs_metrics_and_examples_table(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    run = FakeRun()
    _install_fake_wandb(monkeypatch, run)
    monitor = start_wandb_monitor(_config(), tmp_path, "run-1")
    assert monitor is not None
    metrics = {
        "loss": 2.0,
        "ctc_loss": 1.5,
        "semantic_loss": 0.5,
        "wer": 0.2,
        "cer": 0.1,
        "short_query_wer": float("nan"),
        "short_query_examples": 0,
        "empty_hypothesis_rate": 0.0,
        "repetition_collapse_rate": 0.0,
        "utterance_wer_gte_100_rate": 0.01,
        "invalid_utf8_rate": 0.0,
        "mean_hypothesis_reference_length_ratio": 0.95,
        "truncated_hypothesis_rate": 0.02,
        "catastrophic_failure_rate": 0.01,
        "examples": [["who are you", "who are you"]],
    }

    assert monitor.log_evaluation(metrics, step=100) is None

    payload, kwargs = run.logs[-1]
    assert payload["trainer/global_step"] == 100
    assert payload["eval/wer"] == 0.2
    assert "eval/short_query_wer" not in payload
    assert isinstance(payload["eval/examples"], FakeTable)
    assert kwargs["step"] == 100
