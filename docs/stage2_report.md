# Aether — Stage 2 Technical Report: Speech-to-LLM Connector

**Author:** Manifestro  
**Started:** 2026-09-22  
**Status:** In progress — Phase 0, Phase 1, and R1/R4 preflight complete
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
| Preflight R1/R4 | 4,096-example frozen-Qwen smoke comparison | **Complete; R1 selected** |
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

## 15. R1 larger-subset smoke

Before the revised Phase 2A experiment, `stage2_phase2a_smoke.ipynb` runs a
bounded frozen-Qwen3-4B ratio-1 preflight on a fixed 4,096-example training
subset and 256-example held-out subset. It checks optimization behavior,
speech dependence, scale stability, throughput, memory, and the viability of
the larger-data path. Its result is explicitly not the final answer to the
Phase 2A research question. The actual Phase 2A design will be recorded after
the experiment plan is revised.

The ratio-1 smoke used frozen AetherSpeech and Qwen3-4B, a trainable R1
Connector, 4,096 fixed training examples, 256 fixed held-out examples,
microbatch 2, gradient accumulation 8, and an effective batch size of 16.
It ran for the complete 5,000-step budget.

Training summary:

| Quantity | Result |
|---|---:|
| First logged train loss | `4.51791` |
| Final train loss | `0.47186` |
| Step-0 validation loss | `4.91474` |
| Step-5,000 validation loss | `0.60120` |
| Validation-loss reduction at step 2,000 | `83.90%` |
| Validation-loss reduction at step 5,000 | `87.77%` |
| Final gradient norm before clipping | `18.49` |
| Output scale, start → end | `0.021048 → 0.101612` |
| Final optimizer throughput | `0.698 steps/s` |
| Allocated-memory growth | `+0.029 GB` |

Selected periodic evaluations on the first 64 held-out examples:

| Step | Validation loss | WER | CER |
|---:|---:|---:|---:|
| 2,000 | `0.79139` | `30.77%` | `19.77%` |
| 3,000 | `0.61485` | `22.20%` | `14.09%` |
| 5,000 | `0.60120` | — | — |

The step-5,000 full evaluation on all 256 held-out examples produced:

| Examples | Loss | WER | CER |
|---:|---:|---:|---:|
| 256 | `0.698386` | **27.97%** | **16.33%** |

The difference between the 64-example periodic metric and the full
256-example result is expected; the full result is the authoritative R1
quality number.

Qualitatively, errors were predominantly phonetic or locally linguistic,
not unrelated language-model completions. Examples included `PAY → PAINT`,
`PAPERS → BATHER`, and near-verbatim recovery of long sentences.

## 16. R1 speech-dependence controls

The mandatory controls used 16 held-out examples:

| Condition | WER | CER |
|---|---:|---:|
| Normal speech | **23.68%** | **13.92%** |
| Shuffled speech states | 123.68% | 88.24% |
| Zero speech states | 139.85% | 92.78% |
| Wrong example's speech | 111.28% | 83.55% |
| Truncated speech | 57.89% | 53.46% |

Every perturbation degraded both WER and CER substantially. This rules out
target leakage and generic language-model continuation as explanations for
the normal-path result. It also shows that temporal ordering, example
identity, and the latter portion of the utterance all materially affect the
generated transcript.

**R1 smoke verdict: passed.** Frozen Qwen3-4B can use a learned Connector to
decode unseen AetherSpeech states, and the result depends on the actual
speech input.

## 17. Parallel R4 smoke

The matched companion smoke kept the same 4,096/256 examples, microbatch 2,
effective batch 16, optimizer schedule, evaluation points, Qwen3-4B, and
5,000-step maximum. The only architectural change was enabling the learned
ratio-4 Resampler, reducing the speech-state rate from 12.5 Hz to
approximately 3.125 Hz.

The experiment was stopped after the step-3,000 evaluation because the
comparison was already decisive and `periodic/step_003000.pt` had been
saved.

Matched step-3,000 comparison:

| Metric | R1 | R4 |
|---|---:|---:|
| Train loss | ~`0.53` | `2.68752` |
| Validation loss | **`0.61485`** | `3.01442` |
| WER | **22.20%** | `94.16%` |
| CER | **14.09%** | `70.23%` |
| Throughput | **0.698 steps/s** | 0.601 steps/s |
| Peak allocated VRAM | 11.96 GB | **10.05 GB** |
| Output scale | approaching `0.1016` at completion | `0.09772` |

R4 saved approximately 1.9 GB of peak allocated VRAM but was slower and far
worse on every quality metric. Its hypotheses were grammatical English but
largely unrelated to the reference, for example `HE WAS A MAN OF VERY STRONG
CHARACTER` for `HE ALSO THOUGHT OF HIS MANAGERIAL POSITION`. This is evidence
that the 4× temporal compression removed information needed for accurate
transcription; Qwen then filled the missing evidence with its language prior.

The R4 values above are limited to the confirmed step-3,000 console output;
the complete R4 log was not archived with this report. That limitation does
not affect the architectural decision because the matched gap was already
very large and R4 had also failed to provide a throughput advantage.

**R4 smoke verdict: stopped as a negative result.** Spending the remaining
2,000 steps was not justified.

## 18. R1/R4 decision

R1 is selected for subsequent Stage 2 work:

- it preserves the native 12.5 Hz AetherSpeech representation;
- it reached 27.97% WER / 16.33% CER on the full held-out256 subset after
  seeing only 4,096 training examples;
- its output passed all speech-dependence controls;
- it was faster than the R4 implementation in the matched smoke;
- A100 memory headroom remained sufficient at microbatch 2.

R4 is not carried forward for transcription. Its modest memory saving does
not compensate for the large information loss and lower throughput.

These smoke experiments answer the temporal-rate preflight question, but
they do not close the revised Phase 2A research question or Stage 2 itself.

## 19. Full frozen-R1 plan

The selected next experiment is a full frozen-Qwen R1 baseline on
LibriSpeech `train-clean-100 + train-clean-360`. LoRA is deferred until the
frozen Connector reaches a measured validation plateau. Selective upper
AetherSpeech unfreezing is deferred until a later LoRA run also reaches a
plateau.

The original `karl4th/limmim` repository contains only 20,757 training
rows. This exactly matches the obsolete pre-upsampler Stage 1 cache that
kept 15.7% of the intended 132,553 utterances, despite the dataset card
describing the full `train.100 + train.360` source. It remains valid for
the already completed bounded smoke experiments, but it is rejected as a
source for full training.

`prepare_limmim_v2.ipynb` therefore rebuilds the source from raw
LibriSpeech through frozen Mimi with no duration or CTC-feasibility filter.
It writes exact transcripts, q0 semantic codes, UTF-8 byte targets, source
metadata, and split provenance into restartable Parquet shards. The
expected counts are 132,553 train, 2,703 validation, and 2,620 test.

`stage2_r1_full.ipynb` consumes `karl4th/limmim-v2-en` and initializes the
Connector from the best R1 smoke WER checkpoint but intentionally creates
a new optimizer, scheduler, and step counter. The source path, SHA-256,
source step, and source provenance are stored with the new run. Training
and validation use distinct dataset
splits, and all examples inside each split are cached; the full run does not
reuse the bounded 4,096/256 smoke subsets.

The hard ceiling is 100,000 optimizer steps at effective batch 16, or about
1.6 million example presentations. The notebook selects microbatch 16 with
no accumulation on a 96 GB RTX PRO 6000, 8 × 2 on an 80 GB GPU, and 4 × 4
on a 40 GB GPU. Evaluation runs every 1,000 steps on a fixed 256-example
validation subset. WER is the primary selection and
plateau metric; validation loss and CER remain independently checkpointed.

The plateau rule is fixed before the run:

- significant improvement means at least `0.005` absolute WER;
- plateau counting begins after warmup, at step 3,000;
- three consecutive eligible evaluations without significant improvement
  stop training;
- the stop writes `plateau_report.json`, `plateau_stop.pt`, `last.pt`, and
  `training_summary.json`.

The old output-scale abort threshold of `0.12` is retired because the valid
R1 smoke ended at `0.1016`. The full run warns once above `0.20`, aborts on a
non-finite value, an absolute magnitude above `1.0`, or a single-step change
above `0.05`, and saves `abort_output_scale.pt` on failure. This permits
gradual learned rescaling while retaining protection against discontinuous
failure.

Cached training keeps the unused frozen AetherSpeech encoder on CPU. Each
evaluation uses inference mode, releases temporary generation tensors,
returns unused CUDA allocator blocks after completion, and records allocated
and reserved VRAM before and after cleanup. The notebook also enables
expandable CUDA allocator segments and deletes its diagnostic forward tensors
before entering the long run.
