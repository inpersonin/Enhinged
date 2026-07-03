"""Runtime inference helpers for Enhinged."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from config import DEFAULT_CHECKPOINT_PATH
from model import HinglishGPT, generate, load_model_from_checkpoint


@dataclass
class _RuntimeState:
    model: Optional[HinglishGPT] = None
    encoding: Optional[object] = None
    device: Optional[torch.device] = None
    checkpoint_path: Optional[str] = None


_STATE = _RuntimeState()


def load_model(ckpt_path: str = DEFAULT_CHECKPOINT_PATH, device: Optional[torch.device] = None) -> None:
    """Load a checkpoint into the shared inference runtime."""

    model, encoding, resolved_device = load_model_from_checkpoint(ckpt_path, device)
    _STATE.model = model
    _STATE.encoding = encoding
    _STATE.device = resolved_device
    _STATE.checkpoint_path = ckpt_path


def unload_model() -> None:
    """Release the shared inference model."""

    _STATE.model = None
    _STATE.encoding = None
    _STATE.device = None
    _STATE.checkpoint_path = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_model_loaded() -> bool:
    """Return whether the shared model is currently loaded."""

    return _STATE.model is not None and _STATE.encoding is not None and _STATE.device is not None


def get_loaded_checkpoint_path() -> Optional[str]:
    """Return the active checkpoint path, if any."""

    return _STATE.checkpoint_path


def _build_history_string(history: list[dict]) -> str:
    lines = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def generate_response(
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    seed: Optional[int] = None,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """Generate an Enhinged response from the loaded checkpoint."""

    if not is_model_loaded():
        raise RuntimeError("Model not loaded. Call load_model() before generate_response().")

    model = _STATE.model
    encoding = _STATE.encoding
    device = _STATE.device

    if conversation_history:
        history_text = _build_history_string(conversation_history)
        full_prompt = f"{history_text}\nUser: {prompt}\nAssistant:"
    else:
        full_prompt = f"User: {prompt}\nAssistant:"

    prompt_ids = encoding.encode(full_prompt)
    max_prompt_tokens = max(1, model.config.block_size - max_new_tokens - 10)
    if len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[-max_prompt_tokens:]

    if seed is not None:
        torch.manual_seed(seed)

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
    )

    new_ids = out_ids[0][len(prompt_ids) :].tolist()
    response = encoding.decode(new_ids)

    for stop_phrase in ("User:", "\nUser:", "Assistant:", "\nAssistant:", "<|endoftext|>"):
        if stop_phrase in response:
            response = response[: response.index(stop_phrase)]

    return response.strip()
