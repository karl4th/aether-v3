# Can a Language Model Understand Speech Without a Transcript?

**Aether Research · Stage 2 · Experimental pass complete**

We connected a speech representation directly to a frozen language model and
trained only the small bridge between them. The system learned to transcribe
unseen speech with **16.83% word error rate** on the test set, without passing
a transcript, text tokens, or speech-recognition logits into the language
model. In a clean spoken question, it also recovered the question correctly
and produced the right factual answer, although reliable spoken question
answering remains unfinished.

| Test WER | Test CER | Training speech | Trainable component |
|---:|---:|---:|---|
| **16.83%** | **8.83%** | **~460 hours** | **Speech-to-LLM connector** |

---

## 01 · The Question

Stage 1 showed that a learned speech encoder can recover linguistic structure
from the compact token stream of a frozen neural audio codec. Stage 2 asks the
next question: can a pretrained language model consume that speech
representation directly?

The distinction matters. A conventional voice pipeline first converts speech
to text and then gives the transcript to a language model. That works, but it
forces every later decision through a written intermediate representation.
A direct speech interface could eventually preserve information that a
transcript discards and remove a separately deployed recognition stage.

For this experiment, the language model received continuous speech features
through a learned connector. It never received the reference transcript as an
input.

## 02 · The System

The experimental path was:

```text
audio
  → frozen neural audio codec
  → frozen AetherSpeech encoder
  → trained speech-to-language connector
  → frozen Qwen3-4B
  → text
```

Only the connector and two learned speech-boundary embeddings were trained.
The audio codec, AetherSpeech, and Qwen3-4B remained frozen throughout the
selected experiment. This isolates the research question: whether a small
mapping can make an existing speech representation legible to an existing
language model.

> **Disclosure boundary.** The exact connector architecture, training
> configuration, source repositories, checkpoints, and trained weights are
> proprietary and are not published in this report.

## 03 · Engineering Validation

Before the full experiment, we verified the complete training and generation
path on small controlled runs. The checks covered sequence construction,
attention masks, target labels, finite forward and backward passes, gradient
flow into the connector, frozen-model isolation, cached-state equivalence,
autoregressive generation, checkpoint recovery, and memory stability.

One early issue was especially consequential. The connector initially emitted
vectors at a scale far above the language model's ordinary text embeddings.
Matching those scales before training turned an unstable interface into a
learnable one. The scale remained trainable and converged smoothly during the
accepted runs.

These checks were engineering gates. They established that a later quality
result would measure the proposed interface rather than a broken mask,
detached gradient, invalid cache, or faulty generation loop.

## 04 · Choosing the Speech Rate

We compared two connector inputs: the native temporal rate of the speech
encoder and a version compressed by a factor of four.

The native-rate path learned coherent transcription substantially faster. The
four-times-compressed path produced grammatical language, but much of it was
unrelated to the spoken reference. It also failed to provide a practical
throughput advantage in the matched experiment.

We therefore selected the native-rate representation for full training. The
result suggests that the discarded temporal detail still carried information
needed for accurate word recovery.

## 05 · Full Training

The selected connector was trained on the clean 100-hour and 360-hour
LibriSpeech training subsets, totaling 132,553 utterances. Validation and test
data remained separate from training.

Training stopped after the validation word error rate reached a measured
plateau. Checkpoint selection used validation WER as the primary metric, with
character error rate and validation loss tracked independently. The final
quality numbers below come from complete validation and test splits rather
than the smaller subset used for frequent training-time checks.

## 06 · Transcription Results

Greedy generation was evaluated on every utterance in both held-out splits:

| Split | Utterances | Loss | WER | CER |
|---|---:|---:|---:|---:|
| Validation | 2,703 | 0.4157 | **17.29%** | **9.18%** |
| Test | 2,620 | 0.4123 | **16.83%** | **8.83%** |

Validation and test loss are nearly identical, and test WER is slightly lower
than validation WER. We therefore found no evidence of material overfitting
between these held-out splits.

The Stage 1 diagnostic baseline achieved 16.35% WER and 5.88% CER. Stage 2 is
0.48 absolute WER points behind that baseline on its test split. The direct
language-model path has therefore reached approximately the same word-level
accuracy, but it has not surpassed the dedicated diagnostic decoder.

The remaining errors are usually localized phonetic substitutions, omitted
words, inflection errors, and proper names. They are generally recognizable as
attempts to decode the speech rather than unconstrained language-model prose.

## 07 · Does Meaning Reach the Language Model?

We ran several clean synthetic speech probes through the full audio path. In
one, the speaker asked for the capital of France. The first generated
transcription line matched the spoken sentence exactly, and a direct-answer
prompt produced the correct fact: **Paris**.

This is useful evidence that semantic content can travel through the speech
representation and activate knowledge already present in the language model.
It is still a single diagnostic example, not a semantic benchmark.

Other probes exposed the current weakness. A key named entity could be
misheard while the surrounding sentence remained correct; the language model
would then answer fluently about the wrong entity. Generation also sometimes
repeated the question, emitted formatting tokens, or continued until the token
limit instead of ending cleanly.

The result supports semantic transmission through the interface. It does not
yet establish reliable spoken question answering or instruction following.

## 08 · End-to-End Runtime

The complete audio path was tested with the language model loaded in 4-bit
form. On a 5.8-second clean speech sample, the measured first-run timings on an
NVIDIA RTX PRO 6000 Blackwell Server Edition were:

| Operation | Latency |
|---|---:|
| Speech front end | 0.363 s |
| Transcription generation | 0.597 s |
| Transcription end to end | **0.960 s** |
| Direct-answer generation | 1.573 s |
| Direct answer end to end | **1.937 s** |

Transcription ran at a real-time factor of 0.166, or about six times faster
than the input audio duration. Peak allocated GPU memory was 3.24 GB in this
configuration.

These measurements describe one warmed-up high-end GPU session. They are not
latency claims for consumer hardware, and they do not include a separately
measured time to first token. The direct-answer timing also ended at a fixed
token limit because generation control was not yet reliable.

## 09 · What We Can Conclude

The experiment supports four conclusions:

1. A frozen language model can decode coherent text from frozen AetherSpeech
   states through a small trained connector.
2. The complete live audio path works without relying on precomputed speech
   states at inference time.
3. Direct speech-to-language-model transcription reached 16.83% WER on the
   complete held-out test split.
4. At least one clean spoken question transferred enough meaning for the
   frozen language model to return the correct factual answer.

The experiment does not show reliable open-domain spoken question answering,
robustness to uncontrolled microphones or diverse accents, consistent output
formatting, or better transcription than the Stage 1 baseline. Those remain
open measurements.

## 10 · Current Status

Stage 2 has established its central technical result: a pretrained language
model can consume AetherSpeech's continuous representation directly and
recover both words and, in a preliminary probe, usable meaning without an
intermediate transcript.

The next experimental direction will be chosen after review of these results.

---

*Aether Research · Manifestro · Stage 2 experimental pass complete, 2026 — Architecture, training configuration, repositories, checkpoints, and trained weights are proprietary and are not released.*
