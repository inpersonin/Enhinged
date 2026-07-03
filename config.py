"""Shared configuration for the Enhinged conversational model."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TOKENIZER_NAME = "gpt2"
import os
DEFAULT_CHECKPOINT_PATH = "checkpoints/best.pt"
if not os.path.exists(DEFAULT_CHECKPOINT_PATH) and os.path.exists("best.pt"):
    DEFAULT_CHECKPOINT_PATH = "best.pt"
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "checkpoints"
SUPPORTED_PRETRAINED_MODELS = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")


@dataclass(slots=True)
class GPTConfig:
    """Configuration for the GPT-2 style Enhinged language model."""

    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True
