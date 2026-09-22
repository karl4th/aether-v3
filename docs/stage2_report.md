# Aether — Stage 2 Technical Report: Speech-to-LLM Connector

**Author:** Manifestro  
**Started:** 2026-09-22  
**Status:** In progress — Phase 0 and Phase 1 complete
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
| Phase 1 | Qwen3-4B tiny overfit | **Passed at step 2,000** |
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

## 10. Phase 1 configuration

The accepted Phase 1 run is `run260922-132757`. It used frozen Qwen3-4B,
the frozen Stage 1 encoder, a ratio-1 Connector, 64 fixed training examples,
16 validation examples, batch size 1, and gradient accumulation over 16
microbatches. The maximum budget was 5,000 optimizer steps, but the run was
stopped at the saved step-2,000 checkpoint after the pre-registered tiny
overfit gate passed.

The run used git revision
`ca358de5eb13a2943f0a93b5ad50fe297e14e8ec` and produced the accepted
checkpoint:

```text
/content/drive/MyDrive/aether-v3/stage2/run260922-132757/phase1_2000.pt
```

## 11. Phase 1 embedding calibration

The notebook measured the actual Qwen3-4B text embedding distribution before
the first Connector forward and initialized the FP32 Bridge output scale from
that measurement.

| Statistic | Qwen3-4B text | Connector output |
|---|---:|---:|
| Mean | `-0.0000111` | `0.0000322` |
| Standard deviation | `0.0210481` | `0.0209914` |
| RMS | `0.0210481` | `0.0209914` |
| L2 p50 | `1.06435` | `1.06212` |
| L2 p90 | `1.22907` | `1.06499` |
| L2 p99 | `1.28933` | `1.06621` |

The calibrated initial scale was `0.0210481`. During training it increased
smoothly to `0.0621689` at step 2,000. This was not accompanied by loss,
gradient, or memory instability: the scale remained far below the original broken initialization of 1.0,
while train loss
continued to fall and the final gradient norm fell to 4.90.

## 12. Phase 1 harness checks

| Check | Result |
|---|---|
| Forward loss | `3.98087`, finite |
| Logits shape | `[2, 252, 151936]`, finite |
| Connector gradient norm | `867.68` before clipping |
| Boundary gradients | non-zero |
| Frozen Qwen gradients | `None` |
| Frozen AetherSpeech gradients | `None` |
| Text greedy generation parity | exact token match |
| Live/cache state comparison | `allclose=true`, max diff `0.0009765625` |

The text-generation parity control decoded to ` Paris. The capital of
Germany is Berlin`. Pretraining speech generation completed normally at the
configured token bound; its semantic quality was not a gate.

## 13. Phase 1 training dynamics

The log contains exactly 200 training records for steps 10 through 2,000,
with no duplicate steps or concatenated earlier run.

| Step | Train loss | Gradient norm before clipping | Output scale | Steps/s |
|---:|---:|---:|---:|---:|
| 10 | 4.96514 | 2931.38 | 0.021032 | 0.122 |
| 100 | 3.93283 | 148.41 | 0.021839 | 0.330 |
| 500 | 1.24092 | 104.56 | 0.037832 | 0.386 |
| 1,000 | 0.90545 | 105.40 | 0.050449 | 0.390 |
| 1,500 | 0.08520 | 23.09 | 0.057299 | 0.392 |
| 2,000 | 0.02097 | 4.90 | 0.062169 | 0.395 |

Allocated GPU memory stayed within `8.035–8.100 GB` and ended `0.005 GB`
below its first logged value. Peak allocated memory was `9.737 GB`. No NaN,
Inf, throughput degradation, or memory leak occurred.

Periodic held-out evaluation produced:

| Step | Validation loss | WER | CER |
|---:|---:|---:|---:|
| 0 | 4.81545 | 319.55% | 325.40% |
| 500 | **3.75171** | 122.93% | 107.36% |
| 1,000 | 4.02940 | 145.49% | 109.45% |
| 2,000 | 4.81967 | **106.02%** | **82.83%** |

These held-out numbers are diagnostic only; Phase 1's registered objective
was memorization of the fixed training set.

## 14. Phase 1 full-corpus result and gate decision

The accepted step-2,000 checkpoint was decoded autoregressively over all 64
training and 16 validation examples:

| Split | Examples | Loss | WER | CER |
|---|---:|---:|---:|---:|
| Train | 64 | `0.014056` | **2.12%** | **2.62%** |
| Validation | 16 | `4.807597` | `113.53%` | `87.45%` |

The tiny-overfit gate required train WER at or below 5%. The observed 2.12%
passed with substantial margin, so spending the remaining 3,000-step budget
was unnecessary. Phase 1 therefore closed successfully at step 2,000.

The authoritative result files are `phase1_2000.pt`,
`phase1_2000_corpus_metrics.json`, and `phase1_report.json`. The intermediate
`phase1_5000_corpus_metrics.json` contains the same measurements with the old
planned checkpoint name and is not an accepted result artifact.

## 15. Next experiment

Phase 2A is a bounded frozen-Qwen3-4B ratio-1 transcription probe on a fixed
larger subset. It keeps the Stage 1 encoder and Qwen frozen and trains only
the Connector and speech boundary embeddings. Its own step-0 held-out loss
is the baseline. By step 2,000 it must improve held-out loss by approximately
5% and show speech-dependent decoding; by the hard 5,000-step limit it must
improve held-out loss by approximately 10%, or show a clear sustained WER/CER
trend. The identical subset, order, schedule, initialization policy, and
number of optimizer updates will then be used for the ratio-4 Phase 2B run.
