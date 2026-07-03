"""Prepare Enhinged training data."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np


def clean_text(filepath: str) -> str:
    """Read and normalise a text corpus."""

    with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def encode_text(text: str, encoding_name: str) -> tuple[np.ndarray, str]:
    """Tokenise text with tiktoken and return the chosen dtype."""

    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    start_time = time.time()
    token_ids = encoding.encode_ordinary(text)
    token_ids.append(encoding.eot_token)
    dtype = np.uint16 if encoding.n_vocab <= 65_535 else np.uint32
    array = np.array(token_ids, dtype=dtype)
    elapsed = time.time() - start_time
    print(f"Tokenised {len(array):,} tokens in {elapsed:.2f}s")
    return array, str(array.dtype)


def split_tokens(ids: np.ndarray, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Split token IDs into train and validation segments."""

    val_size = int(len(ids) * val_ratio)
    train_size = len(ids) - val_size
    return ids[:train_size], ids[train_size:]


def save_bin(ids: np.ndarray, filepath: str) -> None:
    """Write token IDs to a binary file."""

    ids.tofile(filepath)
    print(f"Saved {len(ids):,} tokens to {filepath}")


def save_metadata(output_dir: str, encoding_name: str, token_dtype: str, val_ratio: float, total_tokens: int) -> None:
    """Persist preprocessing metadata next to the binary files."""

    metadata = {
        "encoding": encoding_name,
        "token_dtype": token_dtype,
        "val_ratio": val_ratio,
        "total_tokens": total_tokens,
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def prepare_single_file(input_file: str, output_dir: str, val_ratio: float, encoding_name: str) -> None:
    """Prepare a single text corpus file."""

    os.makedirs(output_dir, exist_ok=True)
    text = clean_text(input_file)
    ids, token_dtype = encode_text(text, encoding_name)
    train_ids, val_ids = split_tokens(ids, val_ratio)
    save_bin(train_ids, os.path.join(output_dir, "train.bin"))
    save_bin(val_ids, os.path.join(output_dir, "val.bin"))
    save_metadata(output_dir, encoding_name, token_dtype, val_ratio, len(ids))
    print("Preprocessing complete.")


def prepare_directory_corpus(input_dir: str, output_dir: str, val_ratio: float, encoding_name: str) -> None:
    """Prepare a directory of text files as one combined corpus."""

    os.makedirs(output_dir, exist_ok=True)
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    txt_files = sorted(Path(input_dir).glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    token_chunks: list[np.ndarray] = []
    token_dtype = np.uint16 if encoding.n_vocab <= 65_535 else np.uint32

    for path in txt_files:
        text = clean_text(str(path))
        token_ids = encoding.encode_ordinary(text)
        token_ids.append(encoding.eot_token)
        token_chunks.append(np.array(token_ids, dtype=token_dtype))
        print(f"{path.name}: {len(token_ids):,} tokens")

    ids = np.concatenate(token_chunks)
    train_ids, val_ids = split_tokens(ids, val_ratio)
    save_bin(train_ids, os.path.join(output_dir, "train.bin"))
    save_bin(val_ids, os.path.join(output_dir, "val.bin"))
    save_metadata(output_dir, encoding_name, str(ids.dtype), val_ratio, len(ids))
    print("Preprocessing complete.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Prepare Enhinged conversational data for GPT training.")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="data/")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--encoding", type=str, default="gpt2")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input_file and args.input_dir:
        print("Specify either --input_file or --input_dir, not both.")
        sys.exit(1)

    if args.input_file:
        prepare_single_file(args.input_file, args.output_dir, args.val_ratio, args.encoding)
    elif args.input_dir:
        prepare_directory_corpus(args.input_dir, args.output_dir, args.val_ratio, args.encoding)
    else:
        print("Provide --input_file or --input_dir.")
        sys.exit(1)
