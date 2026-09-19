# AETHER STT v0.1

Speech-to-text: frozen Mimi encoder -> semantic embedding -> `AetherSpeech`
transformer -> CTC head (phase 1, this repo) -> eventually `AetherBridge` +
a frozen Qwen3-4B LM (phase 2, not implemented yet).

```
24kHz speech -> Mimi (frozen, semantic codebook, 12.5Hz, vocab=2048)
             -> Embedding(2048, 768)
             -> AetherSpeech: 8x bidirectional RoPE transformer blocks,
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

Checkpoints, a `config.yaml` snapshot, and a `log.jsonl` land in
`train.output_dir` (see `configs/ctc_base.yaml`). Dev-clean/dev-other WER and
CER are logged every `eval_interval` steps. Set `train.wandb_project` to also
log to Weights & Biases (optional dependency).

## Colab: cross-session caching via Drive

`prepare_cache` downloads ~30GB of raw LibriSpeech audio, but the extracted
result it actually needs to keep (semantic codes + byte targets) is only
~100-150MB. `notebooks/train_ctc.ipynb` mounts Google Drive and, around the
`prepare_cache` call, mirrors just that small extracted cache with
`aether_v3.data.cache_sync` — so a fresh Colab session restores it in
seconds instead of redownloading and re-extracting from scratch. Training
checkpoints/logs (`train.output_dir`) are pointed at Drive directly, and the
notebook auto-resumes from the last checkpoint there if one exists, so a
dropped session doesn't lose the compute units already spent.

Extraction doesn't need a strong GPU (Mimi is small) — the notebook flags
where a T4 runtime is enough, reserving stronger GPU tiers for training.

## Layout

- `src/aether_v3/config.py` — experiment config dataclasses + YAML loader.
- `src/aether_v3/data/` — LibriSpeech loading, byte tokenizer, offline Mimi
  extraction/caching, collate, Drive cache mirroring (`cache_sync.py`).
- `src/aether_v3/models/` — frozen Mimi wrapper, RoPE, `AetherSpeech`
  encoder, CTC head, the combined `AetherCTCModel`.
- `src/aether_v3/training/` — DDP-ready training loop, LR schedule,
  checkpointing.
- `src/aether_v3/eval/` — greedy CTC decode, WER/CER.
- `configs/` — YAML experiment configs.
- `scripts/prepare_data.py` — CLI for the offline extraction step.
