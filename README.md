# AETHER STT v0.1

Speech-to-text: frozen Mimi encoder -> semantic embedding -> `AetherSpeech`
transformer -> CTC head (phase 1, this repo) -> eventually `AetherBridge` +
a frozen Qwen3-4B LM (phase 2, not implemented yet).

```
24kHz speech -> Mimi (frozen, semantic codebook, 12.5Hz, vocab=2048)
             -> Embedding(2048, 768)
             -> AetherSpeech: 8x causal streaming RoPE transformer blocks,
                dim 768, heads 12, FFN 3072
             ├-> CTC head: byte-level UTF-8 + blank
             └-> training-only future-q0 prediction heads (1/2/4 frames)
```

Phase 1 uses a joint objective: CTC measures exact linguistic recovery, while
causal future-q0 prediction prevents transcript supervision from being the only
information preserved by AetherSpeech. Both diagnostic heads are discarded for
the production encoder.

## Setup

```bash
uv sync
```

## 1. Extract semantic codes (once, offline)

Mimi is frozen, so its output is cached to disk instead of recomputed every
epoch.

```bash
# quick smoke test on a handful of utterances first
python scripts/prepare_data.py --config configs/ctc_dummy.yaml --device cpu

# then the real LibriSpeech run (needs a GPU for reasonable speed)
python scripts/prepare_data.py --config configs/ctc_base.yaml --device cuda
```

Also callable from a notebook:

```python
from aether_v3.config import load_config
from aether_v3.data.mimi_cache import prepare_cache

prepare_cache(load_config("configs/ctc_base.yaml"), device="cuda")
```

## 2. Train the CTC branch

Single GPU:

```bash
python -m aether_v3.training.train_ctc --config configs/ctc_base.yaml
```

Multiple GPUs on one node (same code path, no changes needed):

```bash
torchrun --nproc_per_node=4 -m aether_v3.training.train_ctc --config configs/ctc_base.yaml
```

From a notebook:

```python
from aether_v3.config import load_config
from aether_v3.training.train_ctc import run_training

run_training(load_config("configs/ctc_base.yaml"))
```

Every fresh launch creates an immutable directory under `train.runs_dir`, with
resolved config, provenance, JSONL logs, resumable `last.pt`, weights-only model
snapshots, validation reports, and independent best-loss/WER/CER selections.
Use `train.init_encoder_from` for a fresh experiment initialized from an encoder
and `train.resume_run_from` only to continue an interrupted run in place.

The production config requires Weights & Biases monitoring. Put
`WANDB_API_KEY` in the RunPod environment, never in YAML, then launch training
normally. Preflight aborts before the expensive loop if the key is absent or
the W&B run cannot be created. The resulting permanent run and project URLs
are written to `runs/<run-id>/monitoring.json`, so the same private link can be
opened from a phone or laptop. Resume uses the immutable local run ID as the
W&B ID and continues the same charts instead of creating a second run.

Train loss, objective components, learning rate, gradient norm, throughput,
progress and ETA are logged under `train/*` and `progress/*`; validation
WER/CER, failure rates, and reference/hypothesis examples are logged under
`eval/*`. W&B also collects host/GPU telemetry. If remote logging fails after
successful preflight, local JSONL logs and checkpoints continue and the first
failure is recorded in `events.jsonl`.

Optional artifact backups are configured under `artifacts`. Hugging Face reads
the private token from `HF_TOKEN`; Google Drive uses Application Default
Credentials and its optional `google-api-python-client`/`google-auth` packages.
Secrets must not be put in YAML. External upload is disabled unless a destination
is configured.

## Data-cache status

The production Stage 1 config reads `manifestro/stage1_aether` at the immutable
revision recorded in `configs/ctc_base.yaml`. Hugging Face authentication comes
from `HF_TOKEN`; no token is stored in the config or run artifacts. The loader
reads the published Parquet split, validates the required columns, and derives
UTF-8 byte targets lazily from `normalized_text`.

Training batches are bucketed by `semantic_length` and bounded by both example
count and padded semantic-frame budget. The sampler is deterministic per epoch,
partitions complete batches evenly across DDP ranks, and supports exact resume
through the checkpoint's epoch and consumed-batch position. The official test
split is not opened by the training entrypoint.

Validation records overall WER/CER plus duration-based short-query WER and
explicit catastrophic-output rates (empty, repeated, truncated, invalid UTF-8,
and utterance WER at or above 100%).

The legacy LibriSpeech extraction code and `local_arrow` backend remain for
bounded comparison runs; they are not used by the production config.

## Layout

- `src/aether_v3/config.py` — experiment config dataclasses + YAML loader.
- `src/aether_v3/data/` — pinned Loquacious Parquet loading, deterministic
  frame-budget batching, byte tokenizer, legacy Mimi extraction, and collate.
- `src/aether_v3/models/` — frozen Mimi wrapper, RoPE, `AetherSpeech`
  encoder, CTC head, the combined `AetherCTCModel`.
- `src/aether_v3/training/` — DDP-ready training loop, optimizer groups,
  run provenance, checkpointing, metric selections, and optional artifact backup.
- `src/aether_v3/eval/` — greedy CTC decode, WER/CER.
- `configs/` — YAML experiment configs.
- `scripts/prepare_data.py` — CLI for the offline extraction step.
