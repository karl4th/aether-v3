"""Mirrors the (small) extracted Mimi cache to/from a remote directory.

The raw LibriSpeech audio `prepare_cache` downloads is ~30GB and disposable
- it's only ever read once, during extraction. What actually needs to
survive a Colab session (whose local disk resets every time the runtime
recycles) is the *extracted* cache: semantic codes + byte targets, which for
the full train.100+train.360 split is on the order of ~100MB. This module
copies just that between local disk and a remote directory (typically a
mounted Google Drive path) so extraction only has to run once, ever, instead
of once per session.

Deliberately does not duplicate `mimi_cache.py`'s fingerprint-validity
logic: `hydrate_from_remote` only fills in roles missing locally, and
`prepare_cache` (called afterward) still does its own fingerprint check
against the current config, so a stale remote cache is still caught -
not silently trusted just because it came from Drive.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_ROLES = ("train", "validation", "test")


def _read_fingerprint(fingerprint_path: Path) -> str | None:
    if not fingerprint_path.exists():
        return None
    return json.loads(fingerprint_path.read_text()).get("fingerprint")


def hydrate_from_remote(
    local_cache_dir: str | Path,
    remote_cache_dir: str | Path,
    roles: tuple[str, ...] = CACHE_ROLES,
) -> list[str]:
    """Copy any role present in `remote_cache_dir` but missing locally.

    Skips a role that already has a local fingerprint file, regardless of
    its content - `prepare_cache`'s own check decides whether that local
    copy (now possibly restored from remote) is valid for the current
    config; this function's only job is "make it available locally".

    Returns the list of roles actually restored from remote.
    """
    local_cache_dir = Path(local_cache_dir)
    remote_cache_dir = Path(remote_cache_dir)
    if not remote_cache_dir.exists():
        return []

    restored = []
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    for role in roles:
        local_fp_path = local_cache_dir / f"{role}.fingerprint.json"
        if local_fp_path.exists():
            continue

        remote_fp_path = remote_cache_dir / f"{role}.fingerprint.json"
        remote_data_path = remote_cache_dir / role
        if not remote_data_path.exists() or _read_fingerprint(remote_fp_path) is None:
            continue

        logger.info("Restoring '%s' cache from %s ...", role, remote_data_path)
        shutil.copytree(remote_data_path, local_cache_dir / role)
        shutil.copy2(remote_fp_path, local_fp_path)
        restored.append(role)
    return restored


def push_to_remote(
    local_cache_dir: str | Path,
    remote_cache_dir: str | Path,
    roles: tuple[str, ...] = CACHE_ROLES,
) -> list[str]:
    """Copy any role whose local fingerprint differs from (or is absent from) remote.

    Called after `prepare_cache`, so every role with a local fingerprint
    file is known-valid for the current config - safe to publish as-is.

    Returns the list of roles actually pushed to remote.
    """
    local_cache_dir = Path(local_cache_dir)
    remote_cache_dir = Path(remote_cache_dir)

    pushed = []
    for role in roles:
        local_fp_path = local_cache_dir / f"{role}.fingerprint.json"
        local_fp = _read_fingerprint(local_fp_path)
        if local_fp is None:
            continue

        remote_fp_path = remote_cache_dir / f"{role}.fingerprint.json"
        if _read_fingerprint(remote_fp_path) == local_fp:
            continue

        remote_data_path = remote_cache_dir / role
        logger.info("Pushing '%s' cache to %s ...", role, remote_data_path)
        remote_cache_dir.mkdir(parents=True, exist_ok=True)
        if remote_data_path.exists():
            shutil.rmtree(remote_data_path)
        shutil.copytree(local_cache_dir / role, remote_data_path)
        shutil.copy2(local_fp_path, remote_fp_path)
        pushed.append(role)
    return pushed
