"""Qwen3-0.6B prompt/fixed-mask JSONSchemaBench generation, spread over processes.

The per-token loop masks logits in Python, so one process is CPU bound and
leaves the GPU mostly idle. Each worker process loads its own fp16 copy of the
model once (about 2 GiB of VRAM) and takes units from a shared queue; the
notebook sizes the pool with ``HARDWARE.hf_processes()`` (2 on a T4, 6 on an L4).

Prompts are rendered in the kernel (the JSONSchemaBench formatter lives on the
kernel's ``sys.path``); workers only compile, mask, and generate. Every output
file carries an ``identity``: a rerun skips outputs with the same identity and
refuses ones written by a different runtime.
"""

from __future__ import annotations

import multiprocessing
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from .runner import read_identified_json, write_identified_json

MODES = ("prompt", "fixed")
_STATE: dict[str, Any] = {}


@dataclass(frozen=True)
class HfUnit:
    split: str
    unique_id: str
    mode: str  # "prompt" (unconstrained) or "fixed" (FixedMaskController)
    prompt: str
    schema: dict[str, Any]

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")


def output_path(directory: str | Path, unit: HfUnit) -> Path:
    return Path(directory) / f"hf-jsonschema-{unit.split}-{unit.unique_id}-{unit.mode}.json"


def plan_units(rows: Mapping[str, Sequence[Mapping[str, Any]]], render: Callable[[str, dict[str, Any]], str]) -> list[HfUnit]:
    """One unit per (split, item, mode); ``render(split, schema)`` returns the prompt text."""
    import json

    units = []
    for split, items in rows.items():
        for item in items:
            schema = item["json_schema"]
            if isinstance(schema, str):
                schema = json.loads(schema)
            prompt = render(split, schema)
            units.extend(HfUnit(split, str(item["unique_id"]), mode, prompt, schema) for mode in MODES)
    return units


def load_model(model_name: str, token: str | None, torch_threads: int = 1) -> None:
    """Process initializer: one tokenizer, vocabulary, and fp16 CUDA model per worker."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from ..sampling.hf import generate_constrained, generate_unconstrained
    from ..sampling.vocabulary import Vocabulary

    torch.set_num_threads(torch_threads)  # the workers already use every vCPU between them
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, use_fast=False)
    config = AutoConfig.from_pretrained(model_name, token=token)
    vocabulary = Vocabulary.from_qwen_tokenizer(
        tokenizer, model_vocabulary_size=config.vocab_size, eos_token_ids=frozenset({config.eos_token_id}),
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token, torch_dtype=torch.float16).to("cuda").eval()
    _STATE.update(
        model=model, tokenizer=tokenizer, vocabulary=vocabulary, make_controller=fixed_controller,
        constrained=generate_constrained, unconstrained=generate_unconstrained,
    )


def fixed_controller(vocabulary: Any, schema: dict[str, Any]) -> Any:
    from ..grammar import GBNFCompiler
    from ..sampling.automaton import PushdownAutomaton
    from ..sampling.controller import FixedMaskController

    return FixedMaskController(vocabulary, PushdownAutomaton.from_compiled(GBNFCompiler().compile_schema(schema)))


def run_unit(unit: HfUnit, directory: str | Path, identity: Mapping[str, Any], max_new_tokens: int) -> str:
    path = output_path(directory, unit)
    if read_identified_json(path, identity) is not None:
        return "skipped"
    model, tokenizer, vocabulary = _STATE["model"], _STATE["tokenizer"], _STATE["vocabulary"]
    if unit.mode == "fixed":
        result = _STATE["constrained"](
            model, tokenizer, vocabulary, unit.prompt, _STATE["make_controller"](vocabulary, unit.schema),
            max_new_tokens=max_new_tokens, seed=0,
        )
    else:
        result = _STATE["unconstrained"](model, tokenizer, vocabulary, unit.prompt, max_new_tokens=max_new_tokens, seed=0)
    write_identified_json(
        path, {"text": result.text, "token_ids": list(result.token_ids), "stop_reason": result.stop_reason}, identity,
    )
    return "written"


def run_hf_units(
    units: Iterable[HfUnit],
    *,
    directory: str | Path,
    identity: Mapping[str, Any],
    workers: int,
    model_name: str,
    token: str | None,
    max_new_tokens: int = 1024,
    initializer: Callable[..., None] = load_model,
) -> dict[str, int]:
    """Generate every unit not already written; returns counts of ``written`` and ``skipped``."""
    units = list(units)
    pending = [unit for unit in units if read_identified_json(output_path(directory, unit), identity) is None]
    counts: Counter[str] = Counter(skipped=len(units) - len(pending), written=0)
    if not pending:
        return dict(counts)
    work = partial(run_unit, directory=str(directory), identity=dict(identity), max_new_tokens=max_new_tokens)
    if workers <= 1:
        initializer(model_name, token)
        counts.update(work(unit) for unit in pending)
        return dict(counts)
    with ProcessPoolExecutor(
        max_workers=min(workers, len(pending)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initializer,
        initargs=(model_name, token),
    ) as pool:
        # chunksize 1: units differ wildly in length, so hand them out one at a time.
        counts.update(pool.map(work, pending, chunksize=1))
    return dict(counts)


__all__ = ["HfUnit", "fixed_controller", "load_model", "output_path", "plan_units", "run_hf_units", "run_unit"]
