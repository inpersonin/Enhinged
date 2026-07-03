"""Dataset loading and evaluation helpers for Enhinged."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import DEFAULT_TOKENIZER_NAME
from model import generate_text


def load_dataset_metadata(data_dir: str) -> dict:
    """Load optional dataset metadata from data/meta.json."""

    metadata_path = Path(data_dir) / "meta.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class HinglishDataset:
    """Memory-mapped dataset for language-model training."""

    def __init__(
        self,
        data_dir: str,
        block_size: int,
        batch_size: int,
        device: torch.device,
        token_dtype: Optional[str] = None,
    ) -> None:
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        metadata = load_dataset_metadata(data_dir)
        resolved_dtype = np.dtype(token_dtype or metadata.get("token_dtype", "uint16"))

        train_path = os.path.join(data_dir, "train.bin")
        val_path = os.path.join(data_dir, "val.bin")
        self.train_data = np.memmap(train_path, dtype=resolved_dtype, mode="r")
        self.val_data = np.memmap(val_path, dtype=resolved_dtype, mode="r")

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of training or validation sequences."""

        data = self.train_data if split == "train" else self.val_data
        indices = torch.randint(len(data) - self.block_size, (self.batch_size,))

        x = torch.stack(
            [torch.from_numpy(data[index : index + self.block_size].astype(np.int64)) for index in indices]
        )
        y = torch.stack(
            [torch.from_numpy(data[index + 1 : index + self.block_size + 1].astype(np.int64)) for index in indices]
        )

        if self.device.type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset: HinglishDataset,
    device: torch.device,
    n_batches: int = 100,
) -> dict:
    """Return validation metrics for the current model."""

    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.get_batch("val")
            _, loss = model(x, y)
            losses.append(loss.item())

    val_loss = float(np.mean(losses))
    perplexity = float(np.exp(val_loss))

    prompt = "Yeh kya ho raha hai"
    import tiktoken

    encoding = tiktoken.get_encoding(DEFAULT_TOKENIZER_NAME)

    import time

    started_at = time.time()
    for _ in range(5):
        generate_text(model, encoding, prompt, device, max_new_tokens=50)
    elapsed = time.time() - started_at

    token_count = 5 * 50

    tokens_per_second = token_count / max(elapsed, 1e-8)

    results = {
        "val_loss": val_loss,
        "perplexity": perplexity,
        "parameters": model.count_params() if hasattr(model, "count_params") else sum(p.numel() for p in model.parameters()),
        "tokens_per_sec_gen": tokens_per_second,
    }

    for key, value in results.items():
        print(f"  {key:<25}: {value:,.2f}" if isinstance(value, float) else f"  {key:<25}: {value:,}")

    return results
