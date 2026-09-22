"""Restartable LibriSpeech -> Mimi q0 Parquet extraction for limmim-v2."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from aether_v3.config import MimiConfig
from aether_v3.data.tokenizer import text_to_byte_ids
from aether_v3.models.mimi_wrapper import FrozenMimi

logger = logging.getLogger(__name__)


class _LibriAudioRows(torch.utils.data.Dataset):
    def __init__(self, raw: Any, target_sample_rate: int, source_split: str, offset: int) -> None:
        self.raw = raw
        self.target_sample_rate = target_sample_rate
        self.source_split = source_split
        self.offset = offset

    def __len__(self) -> int:
        return len(self.raw) - self.offset

    def __getitem__(self, local_index: int) -> dict[str, Any]:
        source_index = self.offset + local_index
        row = self.raw[source_index]
        audio = row["audio"]
        sample_rate = int(audio["sampling_rate"])
        waveform = np.asarray(audio["array"], dtype=np.float32)
        source_samples = len(waveform)
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                torch.from_numpy(waveform), sample_rate, self.target_sample_rate
            ).numpy()
        return {
            "source_index": source_index,
            "source_split": str(row.get("source_split", self.source_split)),
            "source_sample_rate": sample_rate,
            "source_num_samples": source_samples,
            "waveform": waveform,
            "transcript": str(row["text"]),
            "source_id": str(row.get("id", row.get("file", source_index))),
            "speaker_id": str(row.get("speaker_id", "")),
            "chapter_id": str(row.get("chapter_id", "")),
            "file": str(row.get("file", "")),
        }


def _encode_with_oom_retry(mimi: Any, waveforms: list[np.ndarray]) -> list[np.ndarray]:
    try:
        return mimi.encode_semantic(waveforms)
    except torch.cuda.OutOfMemoryError:
        if len(waveforms) == 1:
            raise
        torch.cuda.empty_cache()
        middle = len(waveforms) // 2
        logger.warning(
            "CUDA OOM on %d waveforms; retrying as %d + %d",
            len(waveforms),
            middle,
            len(waveforms) - middle,
        )
        return _encode_with_oom_retry(mimi, waveforms[:middle]) + _encode_with_oom_retry(
            mimi, waveforms[middle:]
        )


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _write_parquet_atomic(records: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    temporary = path.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pylist(records), temporary, compression="zstd")
    os.replace(temporary, path)


def build_limmim_v2_split(
    raw: Any,
    mimi_config: MimiConfig,
    output_dir: str | Path,
    role: str,
    source_splits: list[str],
    *,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 4,
    shard_size: int = 2048,
    mimi: Any | None = None,
) -> dict[str, Any]:
    """Encode every source row exactly once and save restartable Parquet shards.

    No duration or CTC-feasibility filter is applied. Existing complete shards
    are counted and skipped. A final partial shard is only written together
    with the completion marker, so interrupted runs restart from a safe boundary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "_COMPLETE.json"
    total = len(raw)
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        if int(complete.get("examples", -1)) != total:
            raise RuntimeError(
                f"Completion marker at {complete_path} has {complete.get('examples')} rows, "
                f"but current source has {total}"
            )
        return complete

    shard_paths = sorted(output_dir.glob("part-*.parquet"))
    written = sum(_parquet_rows(path) for path in shard_paths)
    if written > total:
        raise RuntimeError(f"Existing shards contain {written} rows but source has {total}")

    mimi = mimi or FrozenMimi(mimi_config.pretrained_id, mimi_config.num_quantizers, device=device)
    loader = DataLoader(
        _LibriAudioRows(raw, mimi.target_sample_rate, role, written),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    records: list[dict[str, Any]] = []
    shard_index = len(shard_paths)
    progress = tqdm(
        total=total,
        initial=written,
        desc=f"Mimi {role}",
        unit="utt",
        dynamic_ncols=True,
    )

    def flush() -> None:
        nonlocal records, shard_index
        if not records:
            return
        path = output_dir / f"part-{shard_index:06d}.parquet"
        _write_parquet_atomic(records, path)
        shard_index += 1
        records = []

    for batch in loader:
        codes_batch = _encode_with_oom_retry(mimi, [row["waveform"] for row in batch])
        for row, codes in zip(batch, codes_batch, strict=True):
            transcript = row["transcript"]
            byte_target = text_to_byte_ids(transcript)
            if bytes(byte_target).decode("utf-8") != transcript:
                raise RuntimeError(f"Transcript byte round-trip failed for {row['source_id']}")
            semantic_codes = np.asarray(codes, dtype=np.int64).reshape(-1).tolist()
            records.append(
                {
                    "sample_id": f"{role}_{row['source_index']:08d}",
                    "source_id": row["source_id"],
                    "source_split": row["source_split"],
                    "speaker_id": row["speaker_id"],
                    "chapter_id": row["chapter_id"],
                    "file": row["file"],
                    "source_sample_rate": row["source_sample_rate"],
                    "source_num_samples": row["source_num_samples"],
                    "audio_seconds": row["source_num_samples"] / row["source_sample_rate"],
                    "transcript": transcript,
                    "semantic_codes": semantic_codes,
                    "semantic_length": len(semantic_codes),
                    "byte_target": byte_target,
                }
            )
            if len(records) >= shard_size:
                flush()
        progress.update(len(batch))
    flush()
    progress.close()

    final_count = sum(_parquet_rows(path) for path in sorted(output_dir.glob("part-*.parquet")))
    if final_count != total:
        raise RuntimeError(f"Extraction incomplete: saved {final_count}/{total} rows")
    manifest = {
        "format": "limmim-v2",
        "role": role,
        "source_splits": source_splits,
        "examples": final_count,
        "mimi_model_id": mimi_config.pretrained_id,
        "mimi_num_quantizers": mimi_config.num_quantizers,
        "semantic_codebook": 0,
        "semantic_frame_rate_hz": 12.5,
        "filters": [],
    }
    complete_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Mimi %s complete: %d/%d rows", role, final_count, total)
    return manifest
