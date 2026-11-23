"""A pushdown automaton that runs grammar IR over raw bytes.

The grammar IR is compiled into a small instruction program (match a character,
call or return from a rule, fork, jump, accept), in the style of a regex VM with
a call stack. A state is the set of all live configurations (program counter
plus call stack) after the input so far, so ambiguity is handled by tracking
every possibility at once. Call stacks are persistent linked frames, which makes
copying a state cheap; tail calls push no frame, so long strings and plain text
do not grow the stack.

Input arrives as bytes (model tokens are byte strings), and an incremental UTF-8
decoder turns them into codepoints. A token may end in the middle of a
multi-byte character; that state is valid, but the response cannot finish there.

The token-mask engine asks this automaton, for each vocabulary token, "is the
state still viable after these bytes?", which is how the sampler forbids tokens
that would break the grammar. Limits on stack depth, configurations, and
epsilon steps keep a pathological grammar from exhausting memory or time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypeAlias

from .ir import (
    CharSet,
    Choice,
    Empty,
    Expression,
    GrammarProgram,
    Group,
    Literal,
    Repeat,
    RuleRef,
    Sequence,
)


class GrammarExecutionError(ValueError):
    """Base class for grammar compilation and execution failures."""


class InvalidGrammarError(GrammarExecutionError):
    """The grammar cannot be executed (unknown rule, unbounded repeat of something that can be empty)."""


class AutomatonLimitError(GrammarExecutionError):
    """A safety limit (instructions, stack depth, configurations, epsilon steps) was exceeded."""


@dataclass(frozen=True)
class StackFrame:
    """One return address on a persistent (shared, immutable) call stack."""

    return_pc: int
    parent: StackFrame | None
    size: int


@dataclass(frozen=True)
class Configuration:
    """One live thread: where it is in the program and what it will return to."""

    pc: int
    stack: StackFrame | None = None


@dataclass(frozen=True)
class UTF8State:
    """Bytes of a partly read UTF-8 character, and how many it needs in total."""

    pending: tuple[int, ...] = ()
    expected: int = 1

    @property
    def pending_bytes(self) -> bytes:
        return bytes(self.pending)

    def feed(self, byte: int) -> tuple[UTF8State, int | None] | None:
        """Add one byte: the new state and a completed codepoint (or ``None`` if more bytes are needed).

        Returns ``None`` for invalid UTF-8, including overlong forms and surrogates,
        which the lead-byte checks below reject as early as possible.
        """
        if not 0 <= byte <= 255:
            raise ValueError("A byte must be between 0 and 255")
        if not self.pending:
            if byte <= 0x7F:
                return self, byte
            if 0xC2 <= byte <= 0xDF:
                expected = 2
            elif 0xE0 <= byte <= 0xEF:
                expected = 3
            elif 0xF0 <= byte <= 0xF4:
                expected = 4
            else:
                return None
            return UTF8State((byte,), expected), None

        if not 0x80 <= byte <= 0xBF:
            return None
        if len(self.pending) == 1:
            lead = self.pending[0]
            if lead == 0xE0 and byte < 0xA0:
                return None
            if lead == 0xED and byte > 0x9F:
                return None
            if lead == 0xF0 and byte < 0x90:
                return None
            if lead == 0xF4 and byte > 0x8F:
                return None
        pending = (*self.pending, byte)
        if len(pending) < self.expected:
            return UTF8State(pending, self.expected), None
        try:
            character = bytes(pending).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None
        return UTF8State(), ord(character)


@dataclass(frozen=True)
class AutomatonState:
    """Every live configuration after some input, plus any pending UTF-8 bytes. Immutable."""

    configurations: frozenset[Configuration]
    utf8: UTF8State
    byte_offset: int = 0

    @property
    def viable(self) -> bool:
        """Whether any configuration survives, i.e. the input so far is a valid grammar prefix."""
        return bool(self.configurations)

    @property
    def max_stack_depth(self) -> int:
        return max(
            (configuration.stack.size if configuration.stack is not None else 0)
            for configuration in self.configurations
        )


# Instructions of the compiled program.


@dataclass(frozen=True)
class MatchLiteral:
    """Consume exactly this codepoint, then continue at ``target``."""

    codepoint: int
    target: int


@dataclass(frozen=True)
class MatchSet:
    """Consume one codepoint in the character set, then continue at ``target``."""

    characters: CharSet
    target: int


@dataclass(frozen=True)
class Call:
    """Enter a rule. ``return_pc`` is where to resume afterwards; ``None`` is a tail call (no frame)."""

    rule: str
    return_pc: int | None


@dataclass(frozen=True)
class Return:
    """Leave the current rule and resume at the top frame's return address."""


@dataclass(frozen=True)
class Fork:
    """Continue at every target at once (a choice, or a repeat's loop/exit)."""

    targets: tuple[int, ...]


@dataclass(frozen=True)
class Jump:
    """Continue at ``target`` without consuming input."""

    target: int


@dataclass(frozen=True)
class Accept:
    """The root rule finished: the input is a complete sentence of the grammar."""


Instruction: TypeAlias = MatchLiteral | MatchSet | Call | Return | Fork | Jump | Accept


def _nullable(expression: Expression, rules: dict[str, bool]) -> bool:
    # Whether the expression can match the empty string, given which rules can.
    if isinstance(expression, Empty):
        return True
    if isinstance(expression, Literal):
        return not expression.text
    if isinstance(expression, CharSet):
        return False
    if isinstance(expression, RuleRef):
        return rules.get(expression.name, False)
    if isinstance(expression, Group):
        return _nullable(expression.expression, rules)
    if isinstance(expression, Sequence):
        return all(_nullable(item, rules) for item in expression.items)
    if isinstance(expression, Choice):
        return any(_nullable(item, rules) for item in expression.alternatives)
    if isinstance(expression, Repeat):
        return expression.minimum == 0 or _nullable(expression.expression, rules)
    raise TypeError(type(expression))


class _ProgramCompiler:
    """Translate a grammar program into instructions, compiling each expression back to front."""

    def __init__(self, program: GrammarProgram, max_instructions: int) -> None:
        self.program = program
        self.expressions = program.rule_map()
        self.max_instructions = max_instructions
        self.instructions: list[Instruction] = []
        self.rule_entries: dict[str, int] = {}
        self.rule_nullable = {name: False for name in self.expressions}
        self._compute_nullability()

    def _compute_nullability(self) -> None:
        # Fixed-point iteration: rules can refer to each other, so repeat until nothing changes.
        for _ in range(len(self.expressions) + 1):
            updated = {
                name: _nullable(expression, self.rule_nullable)
                for name, expression in self.expressions.items()
            }
            if updated == self.rule_nullable:
                break
            self.rule_nullable = updated

    def add(self, instruction: Instruction) -> int:
        if len(self.instructions) >= self.max_instructions:
            raise AutomatonLimitError(
                f"The grammar exceeds {self.max_instructions} automaton instructions"
            )
        self.instructions.append(instruction)
        return len(self.instructions) - 1

    def compile(self) -> tuple[tuple[Instruction, ...], dict[str, int], int]:
        """The instructions, each rule's entry point, and the start (a call to the root rule)."""
        accept = self.add(Accept())
        for name, expression in self.expressions.items():
            returning = self.add(Return())
            self.rule_entries[name] = self.compile_expression(expression, returning)
        start = self.add(Call(self.program.root, accept))
        return tuple(self.instructions), self.rule_entries, start

    def compile_expression(self, expression: Expression, target: int) -> int:
        """Instructions that match ``expression`` and then continue at ``target``; returns the entry."""
        if isinstance(expression, Empty):
            return self.add(Jump(target))
        if isinstance(expression, Literal):
            entry = target
            for character in reversed(expression.text):
                entry = self.add(MatchLiteral(ord(character), entry))
            return entry
        if isinstance(expression, CharSet):
            return self.add(MatchSet(expression, target))
        if isinstance(expression, RuleRef):
            if expression.name not in self.expressions:
                raise InvalidGrammarError(f"Unknown grammar rule: {expression.name}")
            # Followed directly by a return: a tail call, so no stack frame is needed.
            tail = isinstance(self.instructions[target], Return)
            return self.add(Call(expression.name, None if tail else target))
        if isinstance(expression, Group):
            return self.compile_expression(expression.expression, target)
        if isinstance(expression, Sequence):
            entry = target
            for item in reversed(expression.items):
                entry = self.compile_expression(item, entry)
            return entry
        if isinstance(expression, Choice):
            targets = tuple(
                self.compile_expression(alternative, target)
                for alternative in expression.alternatives
            )
            return self.add(Fork(targets))
        if isinstance(expression, Repeat):
            # x* where x can match nothing would loop forever without consuming input.
            if expression.maximum is None and _nullable(
                expression.expression, self.rule_nullable
            ):
                raise InvalidGrammarError("Unbounded repetition of a nullable expression")
            # Unbounded: the minimum copies, then a loop that forks between another body and the exit.
            if expression.maximum is None:
                loop = self.add(Fork(()))
                body = self.compile_expression(expression.expression, loop)
                self.instructions[loop] = Fork((body, target))
                entry = loop
                for _ in range(expression.minimum):
                    entry = self.compile_expression(expression.expression, entry)
                return entry
            # Bounded: unroll; copies past the minimum are each optional (a fork that may skip ahead).
            entry = target
            for index in reversed(range(expression.maximum)):
                body = self.compile_expression(expression.expression, entry)
                entry = body if index < expression.minimum else self.add(Fork((body, entry)))
            return entry
        raise TypeError(type(expression))


class PushdownAutomaton:
    """Run a compiled grammar over bytes. States are immutable, so they can be cached and shared."""

    def __init__(
        self,
        program: GrammarProgram,
        *,
        max_stack_depth: int = 128,
        max_configurations: int = 4096,
        max_epsilon_steps: int = 100_000,
        max_instructions: int = 20_000,
    ) -> None:
        if min(max_stack_depth, max_configurations, max_epsilon_steps, max_instructions) < 1:
            raise ValueError("Automaton limits must be positive")
        self.program = program
        self.max_stack_depth = max_stack_depth
        self.max_configurations = max_configurations
        self.max_epsilon_steps = max_epsilon_steps
        compiler = _ProgramCompiler(program, max_instructions)
        self.instructions, self.rule_entries, self.start_pc = compiler.compile()
        self._initial = self._closure(frozenset({Configuration(self.start_pc)}))

    @classmethod
    def from_compiled(cls, grammar: Any, **limits: int) -> PushdownAutomaton:
        """Build from a ``CompiledGrammar`` (which keeps the IR it was rendered from)."""
        program = getattr(grammar, "program", None)
        if not isinstance(program, GrammarProgram):
            raise InvalidGrammarError(
                "The compiled grammar has no IR; compile it with GBNFCompiler"
            )
        return cls(program, **limits)

    def initial_state(self) -> AutomatonState:
        return AutomatonState(self._initial, UTF8State())

    # Follow every instruction that consumes no input (jumps, forks, calls, returns) until each
    # configuration waits on a match or accept. A return with an empty stack just ends that thread.
    def _closure(
        self,
        configurations: frozenset[Configuration],
    ) -> frozenset[Configuration]:
        pending = list(configurations)
        visited: set[Configuration] = set()
        steps = 0
        while pending:
            steps += 1
            if steps > self.max_epsilon_steps:
                raise AutomatonLimitError("Epsilon closure exceeded its step limit")
            configuration = pending.pop()
            if configuration in visited:
                continue
            visited.add(configuration)
            if len(visited) > self.max_configurations:
                raise AutomatonLimitError(
                    f"Epsilon closure exceeded {self.max_configurations} configurations"
                )
            instruction = self.instructions[configuration.pc]
            if isinstance(instruction, Jump):
                pending.append(Configuration(instruction.target, configuration.stack))
            elif isinstance(instruction, Fork):
                pending.extend(
                    Configuration(target, configuration.stack)
                    for target in instruction.targets
                )
            elif isinstance(instruction, Call):
                if configuration.stack is None:
                    size = 1
                else:
                    size = configuration.stack.size + 1
                if instruction.return_pc is not None and size > self.max_stack_depth:
                    raise AutomatonLimitError(
                        f"Grammar stack exceeded {self.max_stack_depth} frames"
                    )
                stack = configuration.stack
                if instruction.return_pc is not None:
                    stack = StackFrame(
                        instruction.return_pc,
                        configuration.stack,
                        size,
                    )
                pending.append(Configuration(self.rule_entries[instruction.rule], stack))
            elif isinstance(instruction, Return):
                if configuration.stack is not None:
                    pending.append(
                        Configuration(
                            configuration.stack.return_pc,
                            configuration.stack.parent,
                        )
                    )
        return frozenset(visited)

    # Consume one codepoint in every configuration that can, then close over epsilon moves again.
    def _step(
        self,
        configurations: frozenset[Configuration],
        codepoint: int,
    ) -> frozenset[Configuration]:
        advanced = set()
        for configuration in self._closure(configurations):
            instruction = self.instructions[configuration.pc]
            if isinstance(instruction, MatchLiteral) and instruction.codepoint == codepoint:
                advanced.add(Configuration(instruction.target, configuration.stack))
            elif isinstance(instruction, MatchSet) and instruction.characters.matches(codepoint):
                advanced.add(Configuration(instruction.target, configuration.stack))
        if len(advanced) > self.max_configurations:
            raise AutomatonLimitError(
                f"Transition exceeded {self.max_configurations} configurations"
            )
        return self._closure(frozenset(advanced))

    # The state after ``data``, or None as soon as the bytes leave the grammar (or are invalid UTF-8).
    def advance_bytes(
        self,
        state: AutomatonState,
        data: bytes,
    ) -> AutomatonState | None:
        configurations = state.configurations
        utf8 = state.utf8
        for byte in data:
            decoded = utf8.feed(byte)
            if decoded is None:
                return None
            utf8, codepoint = decoded
            if codepoint is not None:
                configurations = self._step(configurations, codepoint)
                if not configurations:
                    return None
        return AutomatonState(configurations, utf8, state.byte_offset + len(data))

    def can_finish(self, state: AutomatonState) -> bool:
        """Whether the input so far is a complete sentence (never in the middle of a character)."""
        if state.utf8.pending:
            return False
        return any(
            isinstance(self.instructions[configuration.pc], Accept)
            for configuration in self._closure(state.configurations)
        )

    def expected(self, state: AutomatonState) -> tuple[str, ...]:
        """What could come next, for error messages."""
        expected = set()
        for configuration in self._closure(state.configurations):
            instruction = self.instructions[configuration.pc]
            if isinstance(instruction, MatchLiteral):
                expected.add(json.dumps(chr(instruction.codepoint), ensure_ascii=False))
            elif isinstance(instruction, MatchSet):
                expected.add("character set")
        if self.can_finish(state):
            expected.add("end of input")
        return tuple(sorted(expected))

    def describe(self, state: AutomatonState) -> str:
        """A one-line summary of a state, for debugging a dead end."""
        pending = state.utf8.pending_bytes.hex() or "none"
        return (
            f"byte {state.byte_offset}: {len(state.configurations)} configurations, "
            f"maximum stack {state.max_stack_depth}, pending UTF-8 {pending}; "
            f"expected {', '.join(self.expected(state)[:12]) or 'nothing'}"
        )


__all__ = [
    "AutomatonLimitError",
    "AutomatonState",
    "Configuration",
    "GrammarExecutionError",
    "InvalidGrammarError",
    "PushdownAutomaton",
    "StackFrame",
    "UTF8State",
]
