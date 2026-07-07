"""Shared configuration for the Enhinged conversational model."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_TOKENIZER_NAME = "gpt2"

# HF model repository where weights live.
# HF Spaces have a 1 GB storage cap, so model weights are stored separately
# in a dedicated model repo. inference.py downloads best.pt from here
# automatically on first boot if the file isn't already on disk.
HF_MODEL_REPO = "inpersonin/HinGPT"
HF_MODEL_FILENAME = "best.pt"

# Local path where the downloaded checkpoint is cached inside the container.
# /tmp is always writable in HF Spaces (Docker) and is NOT counted toward the
# repo's 1 GB storage limit, making it the ideal cache location.
HF_MODEL_CACHE_PATH = "/tmp/best.pt"

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
