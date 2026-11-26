"""The optional context seam between the episode loop and the model.

Every model-originated request in ``run_episode`` goes through ``ModelSeam``:
agent turns, parse-error follow-ups (the next agent turn after a parse-error
observation), and the optional final paraphrase. Compaction (Tidemark part 4)
uses its own structured-output client and never adds a hidden tool turn here.

With ``context=None`` (the default) the seam is the legacy path, unchanged:
the logical history is sent to ``client.complete`` exactly as before, its
``canonical_hash`` is the journal's ``messages_sha256``, the controller keeps
its own slot check, and no new event is written. Golden journals, manifests,
and episode ids therefore stay byte-identical. Installing Tidemark never turns
this on; a caller has to pass a session.

With a ``ContextSession`` (Tidemark) the session chooses the exact request
messages and measures the rendered prompt; Toolsmith still owns templating,
native tool-call parsing, grammars, validation, and repair. The logical
history stays intact for the journal and ``EpisodeResult.messages``; it is
not assumed to equal the prompt. The journal then records, in order:

    context_render   rendered-prompt hashes, token counts, budgets, dropped ranges
    model_turn       messages_sha256 hashes the rendered messages actually sent
    backend_timings  rendered/evaluated/reused tokens and the turn's timings

A render that cannot fit the required current turn records ``context_render``
with ``status="context_budget_exhausted"`` and the episode aborts with that
code. It never deletes the current request or lowers ``max_tokens``.

A session is per episode and is not shared between concurrent episodes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from toolsmith.evaluation.models import canonical_hash

from .types import ModelClient, ModelTurn

if TYPE_CHECKING:
    from .journal import EpisodeJournal

RENDER_OK = "ok"
CONTEXT_BUDGET_EXHAUSTED = "context_budget_exhausted"
RENDER_STATUSES = (RENDER_OK, CONTEXT_BUDGET_EXHAUSTED)


def context_config_sha256(config: Mapping[str, Any]) -> str:
    """The hash a manifest stores for a canonical context configuration."""
    return canonical_hash(dict(config))


@dataclass(frozen=True)
class RenderedContext:
    """What a session decided to send for one model request, and the evidence behind it.

    ``messages`` and ``tools`` are exactly what goes to ``client.complete``.
    ``dropped_ranges`` are ``[start, end)`` indexes into the logical history.
    Counts are real tokenizer counts of the rendered prompt; ``None`` only when
    the session could not measure them (the journal keeps the null).
    """

    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    strategy: str
    status: str = RENDER_OK
    detail: str | None = None
    epoch: int = 0
    prompt_sha256: str | None = None
    prompt_token_ids_sha256: str | None = None
    rendered_prompt_tokens: int | None = None
    segment_budgets: dict[str, int | None] = field(default_factory=dict)
    segment_tokens: dict[str, int | None] = field(default_factory=dict)
    max_tokens: int | None = None
    slot_headroom: int | None = None
    recitation_sha256: str | None = None
    compaction_pending: bool = False
    dropped_ranges: tuple[tuple[int, int], ...] = ()
    lint: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in RENDER_STATUSES:
            raise ValueError(f"Unknown render status {self.status!r}")

    @property
    def exhausted(self) -> bool:
        """The required current turn cannot fit; nothing may be sent."""
        return self.status == CONTEXT_BUDGET_EXHAUSTED

    @property
    def dropped_messages(self) -> int:
        return sum(end - start for start, end in self.dropped_ranges)

    def message_list(self) -> list[dict[str, Any]]:
        return [dict(message) for message in self.messages]

    def tool_list(self) -> list[dict[str, Any]]:
        return [dict(tool) for tool in self.tools]


class ContextSession(Protocol):
    """What ``run_episode`` needs from a context implementation (Tidemark provides them)."""

    strategy: str
    version: str  # the implementation version, a manifest identity field

    def identity(self) -> dict[str, Any]:
        """The full canonical context configuration (hashed into the manifest identity)."""
        ...

    def render(
        self,
        history: Sequence[Mapping[str, Any]],
        *,
        step: int,
        tools: Sequence[Mapping[str, Any]],
        tool_choice: str | dict[str, Any],
        initial_count: int,
    ) -> RenderedContext:
        """Choose and measure the request for ``history``. ``initial_count`` is the task's opening messages."""
        ...

    def shape_observation(self, message: Mapping[str, Any], *, step: int) -> dict[str, Any]:
        """The tool message to append to the history (part 1's baseline returns it unchanged)."""
        ...

    def backend_timings(self, turn: ModelTurn, rendered: RenderedContext) -> dict[str, Any]:
        """The ``backend_timings`` fields (except ``step``) for one recorded turn."""
        ...


@dataclass(frozen=True)
class ModelCall:
    """One request through the seam: the turn (absent when the render was exhausted) and its evidence."""

    turn: ModelTurn | None
    messages_sha256: str | None
    rendered: RenderedContext | None = None

    @property
    def exhausted(self) -> bool:
        return self.rendered is not None and self.rendered.exhausted


class ModelSeam:
    """The single place ``run_episode`` calls the model. Legacy behavior when ``context`` is None."""

    def __init__(
        self,
        client: ModelClient,
        journal: EpisodeJournal,
        *,
        context: ContextSession | None = None,
        initial_count: int = 0,
    ) -> None:
        self.client = client
        self.journal = journal
        self.context = context
        self.initial_count = initial_count
        self._config_sha256 = context_config_sha256(context.identity()) if context is not None else None

    @property
    def enabled(self) -> bool:
        return self.context is not None

    def summary(self) -> dict[str, Any] | None:
        """The optional ``episode_started.context`` block (None keeps legacy journals unchanged)."""
        if self.context is None:
            return None
        return {
            "strategy": self.context.strategy,
            "context_config_sha256": self._config_sha256,
            "version": self.context.version,
        }

    def call(
        self,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        step: int,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> ModelCall:
        """Render (with a session) and call the model once."""
        if self.context is None:
            # The legacy path, call for call: hash the logical history, send it as is.
            messages_sha256 = canonical_hash(history)
            if tool_choice == "auto":
                turn = self.client.complete(history, tools)
            else:
                turn = self.client.complete(history, tools, tool_choice=tool_choice)
            return ModelCall(turn, messages_sha256)
        rendered = self.context.render(
            history, step=step, tools=tools, tool_choice=tool_choice, initial_count=self.initial_count,
        )
        if rendered.exhausted:
            self._context_render(step, rendered, None)
            return ModelCall(None, None, rendered)
        messages = rendered.message_list()
        messages_sha256 = canonical_hash(messages)
        turn = self.client.complete(messages, rendered.tool_list(), tool_choice=tool_choice)
        return ModelCall(turn, messages_sha256, rendered)

    def record(
        self,
        call: ModelCall,
        turn: ModelTurn,
        *,
        step: int,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> None:
        """Journal the turn: ``context_render``, ``model_turn``, ``backend_timings`` (the first and last with a session)."""
        assert call.messages_sha256 is not None
        if call.rendered is not None:
            self._context_render(step, call.rendered, call.messages_sha256)
        self.journal.model_turn(step=step, turn=turn, messages_sha256=call.messages_sha256, tool_choice=tool_choice)
        if call.rendered is not None:
            assert self.context is not None
            self.journal.backend_timings(step=step, **self.context.backend_timings(turn, call.rendered))

    def shape_observation(self, message: dict[str, Any], *, step: int) -> dict[str, Any]:
        """A tool message on its way into the history; unchanged without a session."""
        if self.context is None:
            return message
        return dict(self.context.shape_observation(message, step=step))

    def _context_render(self, step: int, rendered: RenderedContext, messages_sha256: str | None) -> None:
        last_seq = self.journal.records[-1]["seq"] if self.journal.records else 0
        self.journal.context_render(
            step=step,
            epoch=rendered.epoch,
            strategy=rendered.strategy,
            status=rendered.status,
            detail=rendered.detail,
            context_config_sha256=self._config_sha256,
            messages_sha256=messages_sha256,
            prompt_sha256=rendered.prompt_sha256,
            prompt_token_ids_sha256=rendered.prompt_token_ids_sha256,
            rendered_prompt_tokens=rendered.rendered_prompt_tokens,
            segment_budgets=dict(rendered.segment_budgets),
            segment_tokens=dict(rendered.segment_tokens),
            max_tokens=rendered.max_tokens,
            slot_headroom=rendered.slot_headroom,
            # Part 1 has one epoch, starting at the episode's first record. Compaction (part 4)
            # moves it; history_end_seq is the newest journal record the history reflects.
            epoch_start_seq=0,
            history_end_seq=last_seq,
            recitation_sha256=rendered.recitation_sha256,
            tool_schema_sha256=canonical_hash(rendered.tool_list()),
            compaction_pending=rendered.compaction_pending,
            dropped_ranges=[[start, end] for start, end in rendered.dropped_ranges],
            lint=dict(rendered.lint) or None,
        )


__all__ = [
    "CONTEXT_BUDGET_EXHAUSTED",
    "ContextSession",
    "ModelCall",
    "ModelSeam",
    "RENDER_OK",
    "RENDER_STATUSES",
    "RenderedContext",
    "context_config_sha256",
]
