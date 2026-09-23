"""Optional external backups for completed or interrupted training runs.

Secrets are read by provider SDKs from the environment. They are never fields
in experiment configs, manifests, logs, or checkpoints.
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from aether_v3.config import ArtifactConfig


class ArtifactSyncError(RuntimeError):
    pass


def _files_for_backup(run_dir: Path) -> list[Path]:
    roots = [
        run_dir / "config.resolved.yaml",
        run_dir / "provenance.json",
        run_dir / "status.json",
        run_dir / "train.jsonl",
        run_dir / "events.jsonl",
    ]
    roots.extend((run_dir / "selections").glob("*.json"))
    roots.extend((run_dir / "checkpoints").glob("*.pt"))
    return sorted(path for path in roots if path.is_file())


def sync_run_to_google_drive(run_dir: str | Path, folder_id: str) -> None:
    """Upsert selected run artifacts into ``folder_id/<run_id>`` using ADC."""
    try:
        import google.auth
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        raise ArtifactSyncError(
            "Google Drive sync requires google-api-python-client and google-auth"
        ) from exc

    run_path = Path(run_dir)
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.file"])
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)

    escaped_name = run_path.name.replace("'", "\\'")
    query = (
        f"name = '{escaped_name}' and '{folder_id}' in parents and "
        "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    matches = service.files().list(q=query, fields="files(id)", pageSize=1).execute()["files"]
    if matches:
        run_folder_id = matches[0]["id"]
    else:
        metadata = {
            "name": run_path.name,
            "parents": [folder_id],
            "mimeType": "application/vnd.google-apps.folder",
        }
        run_folder_id = service.files().create(body=metadata, fields="id").execute()["id"]

    for path in _files_for_backup(run_path):
        relative_name = str(path.relative_to(run_path)).replace(os.sep, "__")
        escaped_file = relative_name.replace("'", "\\'")
        file_query = f"name = '{escaped_file}' and '{run_folder_id}' in parents and trashed = false"
        existing = (
            service.files().list(q=file_query, fields="files(id)", pageSize=1).execute()["files"]
        )
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)
        if existing:
            service.files().update(fileId=existing[0]["id"], media_body=media).execute()
        else:
            service.files().create(
                body={"name": relative_name, "parents": [run_folder_id]},
                media_body=media,
                fields="id",
            ).execute()


def publish_checkpoint_to_huggingface(
    run_dir: str | Path,
    checkpoint: str | Path,
    repo_id: str,
    *,
    private: bool,
    selection: str,
) -> None:
    """Publish one selected model snapshot and its run metadata to a model repo."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ArtifactSyncError("HF_TOKEN is required for Hugging Face publication")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    run_path = Path(run_dir)
    remote_root = f"runs/{run_path.name}/{selection}"
    api.upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo=f"{remote_root}/model.pt",
        repo_id=repo_id,
        repo_type="model",
    )
    for name in ("config.resolved.yaml", "provenance.json", "status.json"):
        path = run_path / name
        if path.exists():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"{remote_root}/{name}",
                repo_id=repo_id,
                repo_type="model",
            )


def sync_run_backups(run_dir: str | Path, config: ArtifactConfig) -> list[str]:
    """Run configured backups and return provider errors without hiding them."""
    errors: list[str] = []
    if config.google_drive_folder_id:
        try:
            sync_run_to_google_drive(run_dir, config.google_drive_folder_id)
        except Exception as exc:  # noqa: BLE001 - external provider boundary
            errors.append(f"google_drive: {exc}")
    return errors
