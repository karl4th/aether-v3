# Aether — Stage 2 Technical Report: Speech-to-LLM Connector

**Author:** Manifestro  
**Started:** 2026-09-22  
**Status:** In progress — Phase 0 complete  
**Specification:** `docs/stage2_spec.md`

---

## 1. Research question

Stage 2 tests whether the linguistic representation learned by the frozen
Stage 1 `AetherSpeech` encoder can be translated into the embedding space of
a frozen pretrained language model:

```text
cached AetherSpeech states [T, 768]
  → AetherConnector
  → frozen Qwen
  → text tokens
```

Phase 0 is an engineering gate. It does not measure generalization or final
speech recognition quality. It must prove that the data path, masks, label
construction, gradient flow, generation, KV cache, checkpointing, and
training loop work before a Qwen3-4B experiment is started.

## 2. Experiment ladder and current status

| Phase | Purpose | Status |
|---|---|---|
| Phase 0 | Qwen3-0.6B engineering harness and 128-example memorization gate | **Passed** |
| Phase 1 | Qwen3-4B tiny overfit | Pending |
| Phase 2A | Frozen Qwen3-4B, ratio 1 transcription probe | Pending |
| Phase 2B | Frozen Qwen3-4B, ratio 4 transcription probe | Pending |
| Phase 3+ | LoRA, selective unfreezing, full training, KD and final benchmarks | Gated |

This report is cumulative. Results from later phases will be appended rather
than replacing the Phase 0 evidence.

## 3. Phase 0 configuration

The accepted run is `run260922-130040`.

| Component | Configuration |
|---|---|
| Stage 1 encoder | `manifestro/aetherASR-EN-v0.1`, `last.pt`, frozen |
| Speech state | Native 12.5 Hz AetherSpeech output, hidden size 768 |
| LLM | `Qwen/Qwen3-0.6B`, frozen, BF16 |
| Connector | Ratio 1, Bridge 768 → 3072 → 1024, trainable |
| Boundary embeddings | Trainable `<SPEECH_START>` and `<SPEECH_END>` vectors |
| Task | Direct speech-state-to-transcript generation |
| Dataset | `karl4th/limmim` |
| Data | 128 train examples, 16 validation examples |
| Batch size | 2 |
| Optimizer schedule | AdamW, peak LR `1e-4`, 200-step warmup, cosine decay |
| Training length | 5,000 optimizer steps |
| Gradient clipping | Global norm 1.0 |
| Git revision | `7acbb43a122a674a944ea968320d55b3cafea1af` |

All artifacts were written directly to:

```text
/content/drive/MyDrive/aether-v3/stage2/run260922-130040/
```

The run contains `phase0_5000.pt`, `last.pt`, metric-selected checkpoints,
periodic checkpoints, `log.jsonl`, the frozen configuration, provenance, and
the machine-readable Phase 0 report.

## 4. Engineering harness results

### 4.1 Forward path and sequence construction

A real batch of four cached examples passed through the complete
Connector-to-Qwen path.

| Check | Result |
|---|---|
| Initial loss | `6.89209`, finite |
| Logits | finite, no NaN or Inf |
| Logits shape | `[4, 265, 151936]` |
| Attention masks | matched each assembled sequence length |
| Labels | target positions only; prefix, speech and padding masked with `-100` |

The assembled sequence was verified as:

```text
[text prefix] [SPEECH_START] [speech embeddings] [SPEECH_END] [target + EOS]
```

### 4.2 Gradient flow

One real backward pass produced:

| Quantity | Result |
|---|---:|
| Connector gradient norm before clipping | `76837.80` |
| Speech-start boundary gradient norm | `19625.79` |
| Speech-end boundary gradient norm | `74287.05` |
| Frozen Qwen parameter gradients | `None` |
| Frozen AetherSpeech parameter gradients | `None` |

The large initial raw norms were clipped during optimization. No NaN, Inf,
or late-training gradient instability was observed.

### 4.3 Generation implementation

The text-only control prompt was decoded both with Hugging Face greedy
generation and the project's manual autoregressive KV-cache loop. The first
eight generated token IDs matched exactly and decoded to:

```text
 Paris. The capital of France is also
```

Speech-conditioned generation also terminated correctly at EOS or the
configured `max_new_tokens` bound before training. Output quality at this
point was deliberately not a gate.

### 4.4 Cache integrity

The same semantic Mimi codes were processed through the live frozen Stage 1
encoder and compared with the saved Stage 2 cache.

| Check | Result |
|---|---:|
| Compared shape | `[182, 768]` |
| `torch.allclose(atol=1e-3, rtol=1e-3)` | `true` |
| Maximum absolute difference | `0.0009765625` |

The difference is consistent with the cache's FP16 storage. The cached and
live paths are equivalent within the declared precision.

## 5. Connector embedding-scale correction

The first Phase 0 attempt exposed a scale mismatch: Connector output RMS was
approximately 1.0 while Qwen text embedding RMS was approximately 0.028.
The Connector therefore entered Qwen at roughly 35.7 times the normal text
embedding scale.

Two changes were made before the accepted run:

1. `init_output_scale` was changed from `1.0` to `0.03`.
2. The learned scalar was kept in FP32 while the remaining Connector weights
   ran in BF16. This prevents Adam updates near `1e-5` from rounding away.

The corrected pretraining statistics were:

| Statistic | Qwen text embeddings | Connector output | Ratio |
|---|---:|---:|---:|
| Mean | `-0.0000552` | `-0.0000893` | — |
| Standard deviation | `0.027959` | `0.030030` | 1.074 |
| RMS | `0.027959` | `0.030030` | 1.074 |
| L2 p50 | `0.88796` | `0.96099` | 1.082 |
| L2 p90 | `1.05800` | `0.96366` | 0.911 |
| L2 p99 | `1.14874` | `0.96495` | 0.840 |

The learned scale then moved during training:

| Step | `bridge_output_scale` |
|---:|---:|
| Start | `0.030000` |
| 100 | `0.029746` |
| 500 | `0.036571` |
| 1,000 | `0.045559` |
| 2,000 | `0.050530` |
| 3,000 | `0.051078` |
| 5,000 | `0.050904` |

This confirms both that the initial embedding scale is appropriate and that
the scalar remains trainable under mixed precision.

## 6. Clean 5,000-step memorization run

The accepted `log.jsonl` contains exactly 500 training log entries at
10-step intervals, covering steps 10 through 5,000. There are no duplicate
steps and no concatenated earlier run.

Selected training points:

| Step | Train loss | Gradient norm before clipping | Scale | Steps/s |
|---:|---:|---:|---:|---:|
| 10 | 7.25073 | 29969.49 | 0.029991 | 1.35 |
| 100 | 5.38162 | 1438.44 | 0.029746 | 5.25 |
| 500 | 3.44056 | 30.22 | 0.036571 | 6.27 |
| 1,000 | 1.60812 | 59.05 | 0.045559 | 6.51 |
| 2,000 | 0.14144 | 25.25 | 0.050530 | 6.82 |
| 3,000 | 0.00325 | 0.187 | 0.051078 | 6.93 |
| 4,000 | 0.00249 | 0.180 | 0.050926 | 7.02 |
| 5,000 | 0.00267 | 0.181 | 0.050904 | 7.08 |

Training loss fell by more than three orders of magnitude. Gradients remained
finite and settled without late spikes. Throughput increased during warmup
and remained stable.

Allocated GPU memory changed from `1.8569 GB` at the first logged point to
`1.8648 GB` at the last, a net change of only `+0.0080 GB`. Peak allocated
memory stabilized at approximately `3.69 GB`; no memory leak was detected.

## 7. Evaluation results

Periodic evaluation used the first eight validation examples, as configured
by `eval_max_examples: 8`:

| Step | Validation loss | WER | CER |
|---:|---:|---:|---:|
| 0 | 6.13491 | 100.00% | 95.86% |
| 100 | 5.10439 | 103.17% | 94.53% |
| 500 | **3.62662** | 104.76% | 77.10% |
| 1,000 | 4.21261 | 97.62% | 76.37% |
| 2,000 | 5.73184 | **94.44%** | **75.63%** |
| 3,000 | 6.65965 | 96.83% | 79.03% |
| 4,000 | 6.72064 | 97.62% | 78.73% |
| 5,000 | 6.73308 | 98.41% | 77.25% |

The separate final corpus evaluation used all cached examples:

| Split | Examples | Loss | WER | CER |
|---|---:|---:|---:|---:|
| Train | 128 | `0.002321` | **0.00%** | **0.00%** |
| Validation | 16 | `6.704325` | `139.47%` | `127.92%` |

WER can exceed 100% when the generated hypothesis contains more word errors,
especially insertions, than the reference contains words.

The train/validation divergence is expected for this gate. The experiment
was intentionally constructed to test exact memorization capacity on 128
examples, not generalization. Validation quality will become a decision
metric in the later probe and full-data phases.

## 8. Phase 0 gate decision

**Phase 0 passed.** The evidence establishes that:

- cached Stage 1 states match the live encoder path;
- sequence construction, attention masks and target-only label masking are
  correct;
- gradients reach the Connector and boundary embeddings through frozen Qwen;
- frozen Qwen and AetherSpeech parameters do not accumulate gradients;
- the manual KV-cache generation loop matches Hugging Face greedy decoding;
- embedding scale is aligned with Qwen and remains trainable in mixed
  precision;
- the training loop is numerically and memory stable;
- the Connector can memorize all 128 speech-to-transcript mappings, reaching
  0% train WER and CER under autoregressive generation;
- checkpoints, logs, provenance and machine-readable reports are persisted to
  Google Drive.

The gate supports proceeding to **Phase 1: Qwen3-4B tiny overfit**.

## 9. Limitations

- The result proves engineering correctness and memorization capacity only.
- The 128 examples are far too few to evaluate generalization.
- Validation WER/CER are intentionally poor and are not presented as an ASR
  quality result.
- Qwen3-0.6B was selected for fast harness validation; the intended Stage 2
  model is Qwen3-4B.
- Evaluation uses offline, batch-size-one greedy text generation in a Python
  loop. Its examples-per-second rate is not a streaming latency benchmark.
- English transcription is the only target exercised so far. Semantic
  question answering and speech-native behavior remain later gated work.

## 10. Next experiment

Phase 1 will repeat the tiny-overfit gate with Qwen3-4B while keeping the
Stage 1 encoder frozen, using ratio 1 and training only the Connector and
speech boundary embeddings. Before training, its own Qwen text embedding
statistics must be measured and used to verify the 4B Connector's initial
output scale. Phase 1 must retain the same forward, backward, cache-integrity,
generation, logging, checkpoint and full-train-corpus checks used here.

