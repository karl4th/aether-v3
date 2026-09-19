"""Byte-level UTF-8 "tokenizer" for the CTC branch.

No vocabulary/BPE training needed: text is just its raw UTF-8 bytes (ids
0-255). A single extra id (256) is reserved for the CTC blank symbol, so the
CTC head has vocab_size=257. This makes the CTC branch language-agnostic —
any language expressible in UTF-8 works without touching this file.
"""
from __future__ import annotations

BLANK_ID = 256
VOCAB_SIZE = 257


def text_to_byte_ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def byte_ids_to_text(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8", errors="replace")
