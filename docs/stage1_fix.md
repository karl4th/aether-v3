# Aether Stage 1 Fix — Technical Report

## Status

- Date: 2026-09-23
- Branch: `stage1`
- Scope: rebuild Stage 1 around speaker generalization and online inference
- Target: streaming AetherSpeech with end-to-end system latency goal of 200 ms
- Quality target: WER at or below 15% on diverse unseen voices, without repetition collapse

This report records engineering work completed before integration of the new
private `manifestro/stage1_aether` dataset. It does not claim that the quality or
latency targets have already been reached.

## Problem being corrected

The previous Stage 1 established that AetherSpeech can recover linguistic
content from Mimi semantic tokens on LibriSpeech. Stage 2 also established that
speech-conditioned Qwen can consume AetherSpeech states through a Connector.

The remaining failure is generalization. The old model performed reasonably on
clean audiobook speech but could collapse on an unfamiliar real voice and a
short conversational query:

```text
WHO ARE YOU
→ WH WH WH WH
```

The purpose of the new Stage 1 is therefore not to repeat the original proof of
concept. It is to produce a stronger AetherSpeech that works online and remains
stable across unseen speakers and short real-world utterances.

## Scope and workflow cleanup

- Added `scope.md` with the new Stage 1 experiment contract and acceptance criteria.
- Removed all three legacy notebooks:
  - `notebooks/ctc_upsampler_experiment.ipynb`
  - `notebooks/prepare_data.ipynb`
  - `notebooks/train_ctc.ipynb`
- Training remains callable as Python code. A new public CLI is intentionally
  deferred until the new dataset/cache contract has been integrated.

## Streaming AetherSpeech

The old AetherSpeech encoder used bidirectional self-attention and therefore
could not satisfy an online inference contract. It has been replaced with a
causal streaming encoder while preserving the existing layer parameter names
and dimensions.

Implemented behavior:

- causal self-attention in training and inference;
- no dependency of an emitted state on future semantic frames;
- explicit `AetherSpeechStreamingState` for independent concurrent sessions;
- per-layer projected key/value caches;
- bounded left context, configured by `streaming_left_context_frames`;
- default context of 256 Mimi frames, approximately 20.5 seconds at 12.5 Hz;
- incremental `forward_chunk` without recomputing the full conversation;
- explicit `reset_stream` and `flush_stream` contracts;
- absolute RoPE offsets across successive chunks;
- rejection of padded streaming chunks, preventing invalid time advancement
  for asynchronous sessions;
- training forward avoids creating and detaching inference KV caches.

The implementation was tested for:

- absence of future-to-past information leakage;
- equivalence between full causal forward and chunked forward;
- equivalence across several chunk boundary patterns;
- equivalence after the bounded context window begins rolling;
- bounded KV-cache size;
- reset and flush behavior;
- RoPE offset consistency.

Old bidirectional checkpoints remain structurally loadable because parameter
names were retained. Their historical WER cannot be attributed to the new
causal architecture, so a new matched training run is required.

## Run isolation and provenance

Every fresh training launch now creates an exclusive directory:

```text
runs/<run-id>/
├── config.resolved.yaml
├── provenance.json
├── status.json
├── train.jsonl
├── events.jsonl
├── checkpoints/
├── selections/
├── evaluations/
├── predictions/
└── profiles/
```

The default run ID combines the experiment name, UTC timestamp, and short Git
revision. A caller may supply an explicit ID, but an existing directory is
never silently reused or overwritten.

Recorded provenance currently includes:

- run ID and creation time;
- Git branch and commit;
- dirty-worktree flag and hash of the Git diff;
- hostname and platform;
- Python, PyTorch, and CUDA versions;
- visible CUDA device count;
- fully resolved experiment configuration.

Dataset and Mimi immutable revisions will be included after the final private
dataset release is audited and connected to the loader.

## Initialization and resume contracts

Two previously conflated operations are now separate:

### `init_encoder_from`

Loads only AetherSpeech weights into a fresh experiment. The CTC diagnostic
branch, optimizer, scheduler, counters, sampler position, and run directory are
new. This is the correct operation for continuation-versus-scratch experiments.

### `resume_run_from`

Continues an interrupted run in the same directory and restores:

- full model state;
- optimizer state;
- scheduler state;
- optimizer step;
- epoch and consumed-batch position;
- best metric values;
- Python RNG;
- NumPy RNG;
- Torch CPU RNG;
- CUDA RNG states when available;
- termination reason.

Checkpoint schema and kind are validated so a weights-only snapshot cannot be
mistaken for a resumable training checkpoint.

## Checkpoints and model selection

Checkpoint files are written to a temporary file in the same filesystem and
atomically renamed after successful serialization.

The run keeps two checkpoint classes:

- `checkpoints/last.pt`: complete resumable training state;
- `checkpoints/step_XXXXXXXX_model.pt`: weights-only snapshots retained when a
  validation metric improves.

Selections are small JSON pointers instead of duplicate model files. One
physical snapshot may therefore be selected by several metrics.

Supported independent selection categories are:

- overall WER;
- CER;
- validation loss;
- macro-domain WER;
- short-query WER;
- streaming WER;
- catastrophic-failure rate;
- repetition-collapse rate;
- empty-hypothesis rate.

Only WER, CER, and validation loss are currently computed by the old evaluation
loader. Dataset-aware and catastrophic metrics are registered in the selection
contract but will be calculated only after the new Parquet schema is integrated.
Validation loss is diagnostic and is not the sole source of the final model.

## Optimizer and safety improvements

AdamW construction now provides:

- weight decay for matrix parameters;
- no weight decay for biases, normalization scales, and other vector parameters;
- an independent encoder learning-rate multiplier;
- fused AdamW on compatible CUDA devices;
- `zero_grad(set_to_none=True)` to avoid unnecessary gradient buffer writes.

Training safety now includes:

- non-finite loss detection across distributed ranks;
- gradient norm calculation, clipping, logging, and non-finite detection;
- final resumable checkpoint on normal completion;
- final resumable checkpoint after `SIGINT` or `SIGTERM` at a safe step boundary;
- explicit run states and termination reasons;
- artifact-sync failures recorded as events instead of destroying local state.

## Removal of the CTC-only objective bottleneck

CTC remains useful for exact transcript recovery and WER, but it treats
non-transcribed structure such as breathing, laughter, hesitation, and acoustic
events as blank. It is therefore no longer the only objective applied to
AetherSpeech.

The training model now includes three disposable causal semantic-prediction
heads. From every AetherSpeech state they predict the Mimi q0 token at future
horizons of 1, 2, and 4 semantic frames. The default joint loss is:

```text
L = L_CTC + 0.25 * mean(L_q0@1, L_q0@2, L_q0@4)
```

Properties of the implementation:

- each horizon has its own linear prediction head;
- padding is excluded from both source and target positions;
- horizons longer than a sequence are safely skipped;
- loss remains connected and zero when a batch has no valid prediction pair;
- prediction logits are produced one horizon at a time rather than stacked;
- training logs CTC, semantic-prediction, and combined losses independently;
- validation reports the same three loss values;
- prediction heads are training-only and are not part of the final production
  AetherSpeech encoder.

This objective preserves predictive structure present in Mimi q0; it cannot
recover non-verbal information already discarded by q0. A separate linear-probe
experiment on laughter, breathing, coughs, silence, and other events remains
necessary before claiming that q0 carries those signals. Additional Mimi
streams or a dedicated event/prosody path may still be required for AetherDuplex.

## Progress and logs

Training and validation have separate `tqdm` progress displays. Machine-readable
JSONL logging records:

- step and epoch;
- training loss;
- gradient norm;
- learning rate;
- steps per second;
- examples per second;
- allocated CUDA memory when available;
- validation metrics;
- new-best events;
- completion, interruption, and backup failures.

Accurate audio-hours progress, equivalent epochs, frame-budget throughput, and
audio-based ETA require `audio_seconds` and `semantic_length` from the new
dataset loader and are intentionally deferred until that integration.

In distributed training, validation is now executed only by the main process
instead of redundantly evaluating the complete validation set on every GPU.

## External artifact storage

Optional provider integrations were prepared without embedding credentials:

### Google Drive

- uses Google Application Default Credentials;
- creates or reuses a run-specific folder under a configured parent folder;
- uploads resolved config, provenance, status, JSONL logs, selections, and
  checkpoint files;
- uses resumable media uploads;
- can be requested on completion or interruption.

Google support requires `google-api-python-client` and `google-auth` in the
remote environment. These packages are not mandatory for local training.

### Hugging Face

- reads authentication only from `HF_TOKEN`;
- creates/uses a private model repository by default;
- publishes a selected model snapshot with resolved config and provenance;
- supports explicit publication categories;
- performs no upload unless a repository and selection policy are configured.

Neither provider was contacted during local validation. Network authentication,
real uploads, retry behavior under Pod shutdown, and remote checksum receipts
remain to be tested on an authorized environment.

## LoquaciousSet cache integration

The supplied LoquaciousSet Mimi cache specification provides the minimum fields
needed for the new Stage 1:

- stable sample and source identifiers;
- official train, validation, and reserved test splits;
- `speaker_id` for leakage checks;
- `audio_seconds`;
- original and normalized transcripts;
- Mimi `semantic_codes` and `semantic_length`;
- immutable dataset/model provenance;
- shard checksums, processing statistics, and failure manifests.

The published revision inspected for integration is
`4bb733b62abd021c4a153ff5196682912933588e`. Its manifests report:

- 1,089,960 successful examples and zero recorded failures;
- 2,499.27 train hours in 101 shards;
- 15.78 validation hours in one shard;
- 16.60 reserved test hours in one shard;
- 114,342,135 Mimi q0 tokens in total;
- pinned LoquaciousSet and `kyutai/moshiko-pytorch-bf16` revisions;
- Mimi q0 at 12.5 Hz with vocabulary size 2,048.

The actual validation Parquet schema and all 7,759 validation rows were checked
for semantic length equality, q0 code range, non-empty normalized text, and
manifest totals. A full cross-shard checksum, uniqueness, and speaker-overlap
audit was not run locally; that belongs in the Pod preflight.

The training loader now reads the private repository at the pinned revision,
uses `HF_TOKEN` implicitly, validates required columns, and derives UTF-8 byte
targets lazily from `normalized_text`. An explicit per-split Parquet glob keeps
validation from resolving or downloading train/test shards. The training
entrypoint never opens the reserved test split.

The new deterministic sampler groups similar lengths and limits each batch by
both maximum examples and padded q0-frame budget. Complete batches are divided
evenly between DDP ranks, and `epoch + batches_in_epoch` still reproduces the
exact data position on resume. Logs now report measured examples/second and
audio-hours/hour rather than estimating throughput from a fixed batch size.

Validation now reports the required failure-oriented metrics in addition to
aggregate WER/CER: duration-based 1–5 second short-query WER, empty hypothesis,
four-token consecutive repetition collapse, utterance WER at or above 100%,
invalid UTF-8 replacement, strong truncation, hypothesis/reference length
ratio, and their catastrophic-failure union. Non-finite slice metrics are
excluded from best-checkpoint selection.

## Validation performed

The complete local quality gate passed:

```text
ruff check: passed
ruff format --check: passed
mypy src: passed (30 source files)
pytest: 166 passed
git diff --check: passed
```

A synthetic CPU end-to-end exercise also completed:

1. created tiny train and validation caches;
2. trained for two optimizer steps;
3. created the expected run directory and metric selections;
4. saved a resumable `last.pt`;
5. resumed the same run;
6. continued successfully to step three;
7. produced final `status: completed`.

This validates local contracts only. No CUDA training, performance benchmark,
Google Drive upload, Hugging Face publication, microphone inference, listening
test, or LoquaciousSet training was performed.

A bounded RunPod smoke on an RTX 3090 validated one real validation batch at
commit `c70267c`: 32 examples, 850 q0 frames, maximum length 37 frames, full
joint forward/backward plus fused AdamW step, 0.697 seconds and 1.232 GiB peak
allocated VRAM. This is an execution-contract check, not a quality result or a
safe full-run batch-budget measurement.

The first real resume attempt exposed a CUDA-only checkpoint bug: loading with
`map_location=cuda` also moved the serialized CPU RNG state to CUDA, which
`torch.set_rng_state` rejects. RNG restoration now explicitly moves CPU and
per-device CUDA RNG ByteTensors back to CPU before installing them. A regression
test covers the device-mapped CPU state contract.

## Next phase

Next:

1. validate every remote checksum and schema invariant during Pod preflight;
2. verify sample, source, and known-speaker separation across splits;
3. benchmark and tune the semantic-frame budget on the selected GPU;
4. run a bounded CUDA smoke test;
5. compare initialization from the previous Stage 1 checkpoint against scratch;
6. train and select the best robust streaming AetherSpeech;
7. evaluate a fixed microphone stress set including `WHO ARE YOU` on the
    previously failing unseen voice.

The full research target remains WER at or below 15% on diverse unseen voices,
with streaming inference and without the observed repetition-collapse failure.
