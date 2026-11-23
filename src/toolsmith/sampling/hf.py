"""Run the custom constrained sampler inside Hugging Face ``model.generate``.

``ConstrainedLogitsProcessor`` plugs a mask controller into generation: before
each step it commits the token generated last step, asks the controller for the
mask, and sets every disallowed token's logit to ``-inf``. ``generate_constrained``
and ``generate_unconstrained`` run the same pipeline with and without it, so
their outputs and timings can be compared directly. Greedy decoding (the
default) with a fixed seed makes runs repeatable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .controller import (
    BatchOneSynchronizer,
    FixedMaskController,
    TriggeredMaskController,
    UnsupportedGenerationError,
)
from .vocabulary import Vocabulary


@dataclass(frozen=True)
class GenerationResult:
    """One generation: text, bytes, token ids, why it stopped, timings, and mask statistics."""

    text: str
    raw_bytes: bytes
    token_ids: tuple[int, ...]
    stop_reason: str
    prompt_tokens: int
    elapsed_seconds: float
    mask_seconds: tuple[float, ...]
    activation_step: int | None
    cache_info: dict[str, int]

    @property
    def tokens_per_second(self) -> float:
        return len(self.token_ids) / self.elapsed_seconds if self.elapsed_seconds else 0.0


def _torch() -> Any:
    # torch is imported lazily so the rest of toolsmith works without it.
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "The Hugging Face sampler requires a PyTorch runtime and toolsmith[colab]"
        ) from exc
    return torch


class ConstrainedLogitsProcessor:
    """A ``transformers`` logits processor that applies a controller's mask at every step."""

    def __init__(
        self,
        controller: FixedMaskController | TriggeredMaskController,
    ) -> None:
        self.controller = controller
        self.synchronizer = BatchOneSynchronizer()
        self.torch = _torch()

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """Masked logits for the next token (called by ``generate`` once per step)."""
        # The token chosen at the previous step is only visible now, as the last input id.
        token_id = self.synchronizer.synchronize(input_ids)
        if token_id is not None:
            self.controller.commit(token_id)
        decision = self.controller.mask()
        if len(scores.shape) != 2 or int(scores.shape[0]) != 1:
            raise UnsupportedGenerationError("Logits must have shape [1, vocabulary]")
        width = int(scores.shape[1])
        torch = self.torch
        masked = scores.clone()
        # Active grammar: keep only the allowed ids. Free text: drop only the blocked ids.
        if decision.allowed_token_ids is not None:
            allowed = tuple(
                token_id for token_id in decision.allowed_token_ids if token_id < width
            )
            if not allowed:
                raise RuntimeError("No allowed token exists in the model logits vocabulary")
            keep = torch.zeros(width, dtype=torch.bool, device=scores.device)
            keep[torch.tensor(allowed, dtype=torch.long, device=scores.device)] = True
            masked[0] = masked[0].masked_fill(~keep, float("-inf"))
        else:
            blocked = tuple(
                token_id for token_id in decision.blocked_token_ids if token_id < width
            )
            if blocked:
                masked[
                    0,
                    torch.tensor(blocked, dtype=torch.long, device=scores.device),
                ] = float("-inf")
            # Logit columns past the vocabulary (padding) are never valid tokens.
            if width > self.controller.vocabulary.model_vocabulary_size:
                masked[0, self.controller.vocabulary.model_vocabulary_size :] = float("-inf")
        if not bool(torch.isfinite(masked[0]).any().item()):
            raise RuntimeError("All logits became non-finite after constrained masking")
        return masked


def _synchronize_cuda(torch: Any, device: Any) -> None:
    # GPU work is asynchronous; wait for it so wall-clock timings measure the real work.
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


# The shared generation path; ``controller=None`` is plain, unconstrained generation.
def _generate(
    model: Any,
    tokenizer: Any,
    vocabulary: Vocabulary,
    prompt: str,
    *,
    controller: FixedMaskController | TriggeredMaskController | None,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    seed: int,
    generate_options: dict[str, Any] | None,
) -> GenerationResult:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    torch = _torch()
    try:
        from transformers import LogitsProcessorList
    except ImportError as exc:
        raise ImportError("Install toolsmith[colab] to use the Hugging Face sampler") from exc
    # Beam search, multiple sequences, and speculative decoding would break the controller's
    # one-token-per-step bookkeeping, so they are refused.
    options = dict(generate_options or {})
    for key in ("num_beams", "num_return_sequences"):
        if key in options and options[key] != 1:
            raise UnsupportedGenerationError(f"{key} must be 1")
        options.pop(key, None)
    if options.get("assistant_model") is not None:
        raise UnsupportedGenerationError("Speculative/assistant decoding is unsupported")
    options.pop("assistant_model", None)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = model.device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[1])
    processors = (
        LogitsProcessorList([ConstrainedLogitsProcessor(controller)])
        if controller is not None
        else None
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.eval()
    _synchronize_cuda(torch, device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            num_beams=1,
            num_return_sequences=1,
            logits_processor=processors,
            **options,
        )
    _synchronize_cuda(torch, device)
    elapsed = time.perf_counter() - started
    generated = tuple(int(token) for token in output[0, prompt_tokens:].tolist())
    if controller is not None:
        # The processor never sees the final token (no step follows it), so commit any tokens it
        # missed, then check the grammar is complete.
        processed = controller.generated_ids
        if generated[: len(processed)] != processed:
            raise RuntimeError("Controller/generated token prefixes diverged")
        for token_id in generated[len(processed) :]:
            controller.commit(token_id)
        stop_reason = (
            "eos" if generated and generated[-1] in vocabulary.eos_token_ids else "length"
        )
        controller.finalize(stop_reason)
        raw = controller.generated_bytes()
        mask_seconds = tuple(controller.mask_seconds)
        activation_step = (
            controller.activation_step
            if isinstance(controller, TriggeredMaskController)
            else 0
        )
        cache_info = (
            controller.cache_info()
            if isinstance(controller, TriggeredMaskController)
            else controller.engine.cache_info()
        )
    else:
        stop_reason = (
            "eos" if generated and generated[-1] in vocabulary.eos_token_ids else "length"
        )
        raw = vocabulary.reconstruct_bytes(generated)
        mask_seconds = ()
        activation_step = None
        cache_info = {}
    text = raw.decode("utf-8", errors="strict")  # an incomplete character here is a real failure
    return GenerationResult(
        text,
        raw,
        generated,
        stop_reason,
        prompt_tokens,
        elapsed,
        mask_seconds,
        activation_step,
        cache_info,
    )


def generate_constrained(
    model: Any,
    tokenizer: Any,
    vocabulary: Vocabulary,
    prompt: str,
    controller: FixedMaskController | TriggeredMaskController,
    *,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float | None = None,
    seed: int = 0,
    generate_options: dict[str, Any] | None = None,
) -> GenerationResult:
    """Generate with ``controller`` enforcing its grammar at every step."""
    return _generate(
        model,
        tokenizer,
        vocabulary,
        prompt,
        controller=controller,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        seed=seed,
        generate_options=generate_options,
    )


def generate_unconstrained(
    model: Any,
    tokenizer: Any,
    vocabulary: Vocabulary,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float | None = None,
    seed: int = 0,
    generate_options: dict[str, Any] | None = None,
) -> GenerationResult:
    """Generate the same way with no constraint, as the baseline."""
    return _generate(
        model,
        tokenizer,
        vocabulary,
        prompt,
        controller=None,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        seed=seed,
        generate_options=generate_options,
    )


__all__ = [
    "ConstrainedLogitsProcessor",
    "GenerationResult",
    "generate_constrained",
    "generate_unconstrained",
]
