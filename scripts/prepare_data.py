#!/usr/bin/env python3
"""CLI wrapper for offline Mimi semantic-code extraction.

    python scripts/prepare_data.py --config configs/ctc_base.yaml --device cuda

Also callable from a notebook:

    from aether_v3.config import load_config
    from aether_v3.data.mimi_cache import prepare_cache
    prepare_cache(load_config("configs/ctc_base.yaml"), device="cuda")
"""
from __future__ import annotations

import argparse
import logging

from aether_v3.config import load_config
from aether_v3.data.mimi_cache import prepare_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare_cache(load_config(args.config), device=args.device)


if __name__ == "__main__":
    main()
