"""Deterministic synthetic prompts used only in memory."""

from __future__ import annotations

import hashlib
import random

_WORDS = [
    "amber",
    "bridge",
    "cache",
    "delta",
    "ember",
    "forest",
    "gateway",
    "harbor",
    "index",
    "kernel",
    "local",
    "model",
    "native",
    "orbit",
    "prefix",
    "queue",
    "runtime",
    "signal",
    "token",
    "upstream",
    "vector",
    "window",
]


def stable_prefix(token_target: int, seed: int) -> str:
    """Create a deterministic approximate-token prefix without user data."""
    if token_target < 1:
        raise ValueError("token_target must be positive")
    randomizer = random.Random(seed)
    # Short English words average about 1.3 tokenizer tokens; exact counts are reported by Ollama.
    word_target = max(1, round(token_target / 1.3))
    return " ".join(randomizer.choice(_WORDS) for _ in range(word_target))


def prompt_for(prefix: str, suffix_index: int) -> str:
    return f"{prefix}\nReply with one short sentence. Synthetic case {suffix_index}."


def prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
