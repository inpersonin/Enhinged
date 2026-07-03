"""Model components and sampling utilities for Enhinged."""

from __future__ import annotations

import inspect
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import DEFAULT_TOKENIZER_NAME, GPTConfig, SUPPORTED_PRETRAINED_MODELS


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self._attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, capture_attn: bool = False) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)

        def reshape(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)

        query = reshape(query)
        key = reshape(key)
        value = reshape(value)

        scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        scores = scores.masked_fill(self.bias[:, :, :sequence_length, :sequence_length] == 0, float("-inf"))
        attn = self.attn_dropout(F.softmax(scores, dim=-1))

        if capture_attn:
            self._attn_weights = attn.detach().cpu()

        output = attn @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        return self.resid_dropout(self.c_proj(output))


class MLP(nn.Module):
    """Transformer feed-forward network."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Transformer block with pre-norm residual connections."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, capture_attn: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), capture_attn=capture_attn)
        x = x + self.mlp(self.ln_2(x))
        return x


class HinglishGPT(nn.Module):
    """GPT-2 style language model for conversational responses."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight
        self.apply(self._init_weights)

        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_params(self, exclude_embeddings: bool = False) -> int:
        total = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        if exclude_embeddings:
            total -= self.transformer["wpe"].weight.numel()
        return total

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        capture_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, sequence_length = idx.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block_size {self.config.block_size}"
            )

        positions = torch.arange(sequence_length, device=idx.device)
        token_embeddings = self.transformer["wte"](idx)
        position_embeddings = self.transformer["wpe"](positions)
        x = self.transformer["drop"](token_embeddings + position_embeddings)

        for block in self.transformer["h"]:
            x = block(x, capture_attn=capture_attn)

        x = self.transformer["ln_f"](x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2") -> "HinglishGPT":
        """Load weights from an official GPT-2 checkpoint."""

        if model_type not in SUPPORTED_PRETRAINED_MODELS:
            raise ValueError(f"Unknown model_type: {model_type}")

        from transformers import GPT2LMHeadModel

        config_map = {
            "gpt2": GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=1024),
            "gpt2-medium": GPTConfig(n_layer=24, n_head=16, n_embd=1024, block_size=1024),
            "gpt2-large": GPTConfig(n_layer=36, n_head=20, n_embd=1280, block_size=1024),
            "gpt2-xl": GPTConfig(n_layer=48, n_head=25, n_embd=1600, block_size=1024),
        }

        config = config_map[model_type]
        config.vocab_size = 50257
        config.bias = True

        pretrained = GPT2LMHeadModel.from_pretrained(model_type)
        source_state = pretrained.state_dict()
        model = cls(config)
        target_state = model.state_dict()
        transposed = {"attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"}
        source_keys = [key for key in source_state if not key.endswith(".attn.masked_bias") and not key.endswith(".attn.bias")]

        for source_key in source_keys:
            target_key = source_key
            if target_key not in target_state:
                continue
            with torch.no_grad():
                if any(target_key.endswith(suffix) for suffix in transposed):
                    target_state[target_key].copy_(source_state[source_key].T)
                else:
                    target_state[target_key].copy_(source_state[source_key])

        model.load_state_dict(target_state)
        return model

    def configure_optimiser(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        decay_params = [parameter for parameter in self.parameters() if parameter.requires_grad and parameter.dim() >= 2]
        nodecay_params = [parameter for parameter in self.parameters() if parameter.requires_grad and parameter.dim() < 2]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=betas,
            fused=fused_available if fused_available else False,
        )

    def get_attention_weights(self) -> list[torch.Tensor]:
        """Return attention weights captured during the last forward pass."""

        weights: list[torch.Tensor] = []
        for block in self.transformer["h"]:
            if block.attn._attn_weights is not None:
                weights.append(block.attn._attn_weights)
        return weights


GPT = HinglishGPT


@torch.no_grad()
def generate(
    model: HinglishGPT,
    idx: torch.Tensor,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    do_sample: bool = True,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Generate tokens autoregressively from a prompt."""

    model.eval()

    for _ in range(max_new_tokens):
        context = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size :]
        logits, _ = model(context)
        logits = logits[:, -1, :]

        if repetition_penalty != 1.0:
            for batch_index in range(idx.size(0)):
                unique_tokens = idx[batch_index].unique()
                logits[batch_index, unique_tokens] /= repetition_penalty

        logits = logits / temperature

        if top_k is not None:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            logits[logits < values[:, [-1]]] = float("-inf")

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_to_remove = cumulative_probs > top_p
            sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()
            sorted_to_remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_to_remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)

        probabilities = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1) if do_sample else torch.argmax(probabilities, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return idx


def generate_text(
    model: HinglishGPT,
    enc: tiktoken.Encoding,
    prompt: str,
    device: torch.device,
    **kwargs,
) -> str:
    """Convert a text prompt into generated text."""

    model.eval()
    token_ids = enc.encode(prompt)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    out = generate(model, idx, **kwargs)
    return enc.decode(out[0].tolist())


def _resolve_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_checkpoint(
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> tuple[HinglishGPT, tiktoken.Encoding, torch.device]:
    """Load a trained Enhinged checkpoint."""

    resolved_device = _resolve_device(device)
    checkpoint = torch.load(ckpt_path, map_location=resolved_device)
    model = HinglishGPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(resolved_device)
    model.eval()
    import tiktoken

    encoding = tiktoken.get_encoding(DEFAULT_TOKENIZER_NAME)
    return model, encoding, resolved_device
