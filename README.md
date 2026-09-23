# AETHER STT v0.1

Speech-to-text: frozen Mimi encoder -> semantic embedding -> `AetherSpeech`
transformer -> CTC head (phase 1, this repo) -> eventually `AetherBridge` +
a frozen Qwen3-4B LM (phase 2, not implemented yet).

```
24kHz speech -> Mimi (frozen, semantic codebook, 12.5Hz, vocab=2048)
             -> Embedding(2048, 768)
             -> AetherSpeech: 8x causal streaming RoPE transformer blocks,
                dim 768, heads 12, FFN 3072
             -> CTC head: Linear(768, 257), byte-level UTF-8 + blank
```

Phase 1 goal: get the CTC branch recognizing English speech decently on its
own (LibriSpeech) before wiring up the LM branch.

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

Optional artifact backups are configured under `artifacts`. Hugging Face reads
the private token from `HF_TOKEN`; Google Drive uses Application Default
Credentials and its optional `google-api-python-client`/`google-auth` packages.
Secrets must not be put in YAML. External upload is disabled unless a destination
is configured.

## Data-cache status

`prepare_cache` downloads ~30GB of raw LibriSpeech audio, but the extracted
result it actually needs to keep (semantic codes + byte targets) is only
~100-150MB, and extraction is CPU-decode-bound - it doesn't benefit from a
strong GPU. So the notebook workflow is split in two:

The old notebook workflow has been removed. Data-cache design is intentionally
pending the contract audit for the new private Stage 1 dataset. Training itself
is a normal Python process suitable for a persistent GPU Pod.

Re-running `prepare_data.ipynb` is only needed if `configs/ctc_base.yaml`'s
data settings change - a mismatched fingerprint makes `prepare_cache` raise
rather than silently reusing a stale cache.

## Layout

- `src/aether_v3/config.py` — experiment config dataclasses + YAML loader.
- `src/aether_v3/data/` — LibriSpeech loading, byte tokenizer, offline Mimi
  extraction/caching, collate, Drive cache mirroring (`cache_sync.py`).
- `src/aether_v3/models/` — frozen Mimi wrapper, RoPE, `AetherSpeech`
  encoder, CTC head, the combined `AetherCTCModel`.
- `src/aether_v3/training/` — DDP-ready training loop, optimizer groups,
  run provenance, checkpointing, metric selections, and optional artifact backup.
- `src/aether_v3/eval/` — greedy CTC decode, WER/CER.
- `configs/` — YAML experiment configs.
- `scripts/prepare_data.py` — CLI for the offline extraction step.
