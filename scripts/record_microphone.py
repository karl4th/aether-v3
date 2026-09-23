"""Record one mono WAV utterance from the default microphone until Enter.

Run from the repository root:

    uv run --with sounddevice python scripts/record_microphone.py

Use ``--list-devices`` to inspect available inputs and ``--device`` to select
one by index or name. The default output is a timestamped file under
``recordings/``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
except ImportError as error:
    raise SystemExit(
        "Missing sounddevice. Run with: "
        "uv run --with sounddevice python scripts/record_microphone.py"
    ) from error


def _default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("recordings") / f"question-{stamp}.wav"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Output WAV path")
    parser.add_argument("--sample-rate", type=int, default=24_000)
    parser.add_argument("--device", default=None, help="Input device index or name")
    parser.add_argument("--list-devices", action="store_true")
    return parser


def _input_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    args = _parser().parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    if args.sample_rate <= 0:
        raise SystemExit("--sample-rate must be positive")

    output = args.output or _default_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[np.ndarray] = []

    def callback(
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time_info
        if status:
            print(f"Microphone status: {status}", file=sys.stderr, flush=True)
        chunks.append(indata.copy())

    device = _input_device(args.device)
    print("Recording started. Speak now, then press ENTER to stop.", flush=True)
    try:
        with sd.InputStream(
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
            device=device,
            callback=callback,
        ):
            input()
    except KeyboardInterrupt:
        print("\nStopping recording...", flush=True)
    except Exception as error:
        raise SystemExit(
            f"Could not open the microphone: {error}\n"
            "Check OS microphone permission or run with --list-devices."
        ) from error

    if not chunks:
        raise SystemExit("No audio was captured")

    waveform = np.concatenate(chunks, axis=0).reshape(-1)
    sf.write(output, waveform, args.sample_rate, subtype="PCM_16")
    duration = len(waveform) / args.sample_rate
    peak = float(np.abs(waveform).max(initial=0.0))
    print(
        f"Saved: {output.resolve()}\n"
        f"Duration: {duration:.2f} s | Sample rate: {args.sample_rate} Hz | Peak: {peak:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
