import json
from pathlib import Path

from aether_v3.data.cache_sync import hydrate_from_remote, push_to_remote


def _write_role(cache_dir: Path, role: str, fingerprint: str, content: str = "data") -> None:
    role_dir = cache_dir / role
    role_dir.mkdir(parents=True, exist_ok=True)
    (role_dir / "payload.txt").write_text(content)
    (cache_dir / f"{role}.fingerprint.json").write_text(json.dumps({"fingerprint": fingerprint}))


def test_hydrate_restores_role_missing_locally(tmp_path):
    local_dir = tmp_path / "local"
    remote_dir = tmp_path / "remote"
    _write_role(remote_dir, "train", "abc123")

    restored = hydrate_from_remote(local_dir, remote_dir)

    assert restored == ["train"]
    assert (local_dir / "train" / "payload.txt").read_text() == "data"
    assert json.loads((local_dir / "train.fingerprint.json").read_text())["fingerprint"] == "abc123"


def test_hydrate_skips_role_already_present_locally(tmp_path):
    local_dir = tmp_path / "local"
    remote_dir = tmp_path / "remote"
    _write_role(local_dir, "train", "local-fp", content="local-data")
    _write_role(remote_dir, "train", "remote-fp", content="remote-data")

    restored = hydrate_from_remote(local_dir, remote_dir)

    assert restored == []
    assert (local_dir / "train" / "payload.txt").read_text() == "local-data"


def test_hydrate_skips_role_missing_from_remote(tmp_path):
    local_dir = tmp_path / "local"
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()

    restored = hydrate_from_remote(local_dir, remote_dir)

    assert restored == []
    assert not (local_dir / "train").exists()


def test_hydrate_no_op_when_remote_missing(tmp_path):
    local_dir = tmp_path / "local"
    remote_dir = tmp_path / "remote"

    restored = hydrate_from_remote(local_dir, remote_dir)

    assert restored == []
    assert not local_dir.exists()


def test_push_publishes_role_missing_from_remote(tmp_path):
    local_dir = tmp_path / "local"
    remote_dir = tmp_path / "remote"
    _write_role(local_dir, "train", "abc123")

    pushed = push_to_remote(local_dir, remote_dir)

    assert pushed == ["train"]
    assert (remote_dir / "train" / "payload.txt").read_text() == "data"
    assert (
        json.loads((remote_dir / "train.fingerprint.json").read_text())["fingerprint"] == "abc123"
    )


def test_push_skips_role_already_matching_remote(tmp_path):
    local_dir = tmp_path / "local"
    remote_dir = tmp_path / "remote"
    _write_role(local_dir, "train", "same-fp")
    _write_role(remote_dir, "train", "same-fp")

    pushed = push_to_remote(local_dir, remote_dir)

    assert pushed == []


def test_push_overwrites_role_with_stale_remote_fingerprint(tmp_path):
    local_dir = tmp_path / "local"
    remote_dir = tmp_path / "remote"
    _write_role(local_dir, "train", "new-fp", content="new-data")
    _write_role(remote_dir, "train", "old-fp", content="old-data")

    pushed = push_to_remote(local_dir, remote_dir)

    assert pushed == ["train"]
    assert (remote_dir / "train" / "payload.txt").read_text() == "new-data"


def test_push_skips_role_without_local_fingerprint(tmp_path):
    local_dir = tmp_path / "local"
    remote_dir = tmp_path / "remote"
    local_dir.mkdir()

    pushed = push_to_remote(local_dir, remote_dir)

    assert pushed == []
    assert not remote_dir.exists()
