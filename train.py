"""Training and evaluation entry point for Enhinged."""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import time
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config import DEFAULT_OUTPUT_DIR, GPTConfig
from model import HinglishGPT, load_model_from_checkpoint
from utils import HinglishDataset, evaluate_model


class Trainer:
    """Train an Enhinged checkpoint."""

    def __init__(
        self,
        model: HinglishGPT,
        dataset: HinglishDataset,
        out_dir: str,
        learning_rate: float = 6e-4,
        weight_decay: float = 1e-1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        grad_clip: float = 1.0,
        warmup_iters: int = 200,
        lr_decay_iters: int = 5000,
        min_lr: float = 6e-5,
        max_iters: int = 5000,
        batch_size: int = 32,
        gradient_accum_steps: int = 1,
        eval_interval: int = 200,
        eval_iters: int = 50,
        device: Optional[torch.device] = None,
        dtype: str = "bfloat16",
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model = model
        self.dataset = dataset
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.gradient_accum = gradient_accum_steps
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = self.device.type

        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        torch_dtype = dtype_map[dtype]
        self.scaler = torch.cuda.amp.GradScaler(enabled=dtype == "float16" and self.device_type == "cuda")
        self.autocast_ctx = (
            torch.amp.autocast(device_type=self.device_type, dtype=torch_dtype)
            if self.device_type == "cuda"
            else nullcontext()
        )
        self.optimiser = model.configure_optimiser(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=(beta1, beta2),
            device_type=self.device_type,
        )

        self.iter_num = 0
        self.best_val = float("inf")

    def get_lr(self, iteration: int) -> float:
        if iteration < self.warmup_iters:
            return self.learning_rate * iteration / max(1, self.warmup_iters)
        if iteration >= self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (iteration - self.warmup_iters) / max(1, self.lr_decay_iters - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        self.model.eval()
        losses: dict[str, float] = {}
        for split in ("train", "val"):
            split_losses = torch.zeros(self.eval_iters)
            for index in range(self.eval_iters):
                x, y = self.dataset.get_batch(split)
                with self.autocast_ctx:
                    _, loss = self.model(x, y)
                split_losses[index] = loss.item()
            losses[split] = split_losses.mean().item()
        self.model.train()
        return losses

    def save_checkpoint(self, tag: str = "latest") -> str:
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
            "model_config": dataclasses.asdict(self.model.config),
            "iter_num": self.iter_num,
            "best_val": self.best_val,
        }
        path = os.path.join(self.out_dir, f"{tag}.pt")
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state"])
        self.iter_num = checkpoint["iter_num"]
        self.best_val = checkpoint["best_val"]

    def train(self) -> None:
        self.model.train()
        self.model.to(self.device)
        tokens = 0
        start_time = time.time()

        while self.iter_num < self.max_iters:
            learning_rate = self.get_lr(self.iter_num)
            for param_group in self.optimiser.param_groups:
                param_group["lr"] = learning_rate

            if self.iter_num % self.eval_interval == 0:
                losses = self.estimate_loss()
                validation_perplexity = math.exp(losses["val"])
                print(
                    f"iter {self.iter_num:5d} | train loss {losses['train']:.4f} | "
                    f"val loss {losses['val']:.4f} | val ppl {validation_perplexity:.2f} | lr {learning_rate:.2e}"
                )
                self.save_checkpoint("latest")
                if losses["val"] < self.best_val:
                    self.best_val = losses["val"]
                    self.save_checkpoint("best")

            self.optimiser.zero_grad(set_to_none=True)

            for _ in range(self.gradient_accum):
                x, y = self.dataset.get_batch("train")
                with self.autocast_ctx:
                    _, loss = self.model(x, y)
                    loss = loss / self.gradient_accum
                tokens += x.numel()
                self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimiser)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimiser)
            self.scaler.update()
            self.iter_num += 1

            if self.iter_num % 50 == 0:
                elapsed = time.time() - start_time
                print(f"iter {self.iter_num:5d} | loss {loss.item() * self.gradient_accum:.4f} | {tokens / max(elapsed, 1e-8):,.0f} tok/s")
                tokens = 0
                start_time = time.time()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate Enhinged.")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--init_from", choices=["scratch", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], default="scratch")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--lr_decay_iters", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        if args.init_from == "scratch":
            model = HinglishGPT(
                GPTConfig(
                    block_size=args.block_size,
                    vocab_size=50257,
                    n_layer=args.n_layer,
                    n_head=args.n_head,
                    n_embd=args.n_embd,
                    dropout=args.dropout,
                )
            )
        else:
            model = HinglishGPT.from_pretrained(args.init_from)

        model.to(device)
        dataset = HinglishDataset(
            data_dir=args.data_dir,
            block_size=model.config.block_size,
            batch_size=args.batch_size,
            device=device,
        )
        trainer = Trainer(
            model=model,
            dataset=dataset,
            out_dir=args.out_dir,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_iters=args.warmup_iters,
            lr_decay_iters=args.lr_decay_iters,
            max_iters=args.max_iters,
            batch_size=args.batch_size,
            gradient_accum_steps=args.grad_accum_steps,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
            device=device,
            dtype=args.dtype,
            seed=args.seed,
        )

        if args.ckpt_path:
            trainer.load_checkpoint(args.ckpt_path)
        trainer.train()
        return

    ckpt_path = args.ckpt_path or os.path.join(args.out_dir, "best.pt")
    model, _, device = load_model_from_checkpoint(ckpt_path, device)
    dataset = HinglishDataset(
        data_dir=args.data_dir,
        block_size=model.config.block_size,
        batch_size=args.batch_size,
        device=device,
    )
    evaluate_model(model, dataset, device)


if __name__ == "__main__":
    main()
