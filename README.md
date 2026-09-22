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

## Colab: two notebooks, cross-session caching via Drive

`prepare_cache` downloads ~30GB of raw LibriSpeech audio, but the extracted
result it actually needs to keep (semantic codes + byte targets) is only
~100-150MB, and extraction is CPU-decode-bound - it doesn't benefit from a
strong GPU. So the notebook workflow is split in two:

- `notebooks/stage1/prepare_data.ipynb` — run once on a cheap **T4** runtime.
  Downloads LibriSpeech, runs `prepare_cache`, and mirrors just the small
  extracted cache (not the raw audio) to Google Drive via
  `aether_v3.data.cache_sync`.
- `notebooks/stage1/train_ctc.ipynb` — run on a stronger GPU (A100/L4). Restores
  that cache from Drive in seconds (raises with a clear message if it's
  missing, rather than silently re-extracting on the expensive tier) and
  trains. Checkpoints/logs (`train.output_dir`) are written straight to
  Drive, and training auto-resumes from the last checkpoint there if one
  exists, so a dropped session doesn't lose the compute units already
  spent.

Re-running `prepare_data.ipynb` is only needed if `configs/ctc_base.yaml`'s
data settings change - a mismatched fingerprint makes `prepare_cache` raise
rather than silently reusing a stale cache.

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

## Stage 2 notebooks

Stage 2 uses the frozen Stage 1 encoder from the private
`manifestro/aetherASR-EN-v0.1` repository. In Colab, add its access token as
the private secret `HF_TOKEN`.

Run these notebooks in order:

1. `notebooks/stage2/stage2_phase0.ipynb` — complete mandatory Phase 0 engineering
   harness with Qwen3-0.6B: embedding statistics, forward/backward and mask
   checks, text-only KV-cache parity, speech generation, live-vs-cache
   integrity, and a clean 5,000-step memorization run on 128 examples.
2. `notebooks/stage2/stage2_phase1.ipynb` — Phase 1 tiny overfit on exactly 64
   transcription examples with Qwen3-4B. Run only after Phase 0 passes.
3. `notebooks/stage2/stage2_phase2a_smoke.ipynb` — bounded R1 preflight on
   a fixed 4,096-example training subset and 256-example held-out subset
   with frozen Qwen3-4B. This is a smoke experiment before the revised,
   separate Phase 2A experiment. Run only after Phase 1 passes.
4. `notebooks/stage2/stage2_phase2a_2.ipynb` — parallel matched R4 smoke on
   the same 4,096/256 examples, effective batch 16, schedule, and evaluation
   points. It changes only the Connector resampling path from ratio 1 to 4.

Each run writes directly to Google Drive under
`aether-v3/stage2/runYYMMDD-HHMMSS`, including logs, `last.pt`, periodic
checkpoints, and best checkpoints for validation loss, WER, and CER.
