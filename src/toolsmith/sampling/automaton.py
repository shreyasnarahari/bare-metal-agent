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
    pass


class InvalidGrammarError(GrammarExecutionError):
    pass


class AutomatonLimitError(GrammarExecutionError):
    pass


@dataclass(frozen=True)
class StackFrame:
    return_pc: int
    parent: StackFrame | None
    size: int


@dataclass(frozen=True)
class Configuration:
    pc: int
    stack: StackFrame | None = None


@dataclass(frozen=True)
class UTF8State:
    pending: tuple[int, ...] = ()
    expected: int = 1

    @property
    def pending_bytes(self) -> bytes:
        return bytes(self.pending)

    def feed(self, byte: int) -> tuple[UTF8State, int | None] | None:
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
    configurations: frozenset[Configuration]
    utf8: UTF8State
    byte_offset: int = 0

    @property
    def viable(self) -> bool:
        return bool(self.configurations)

    @property
    def max_stack_depth(self) -> int:
        return max(
            (configuration.stack.size if configuration.stack is not None else 0)
            for configuration in self.configurations
        )


@dataclass(frozen=True)
class MatchLiteral:
    codepoint: int
    target: int


@dataclass(frozen=True)
class MatchSet:
    characters: CharSet
    target: int


@dataclass(frozen=True)
class Call:
    rule: str
    return_pc: int | None


@dataclass(frozen=True)
class Return:
    pass


@dataclass(frozen=True)
class Fork:
    targets: tuple[int, ...]


@dataclass(frozen=True)
class Jump:
    target: int


@dataclass(frozen=True)
class Accept:
    pass


Instruction: TypeAlias = MatchLiteral | MatchSet | Call | Return | Fork | Jump | Accept


def _nullable(expression: Expression, rules: dict[str, bool]) -> bool:
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
    def __init__(self, program: GrammarProgram, max_instructions: int) -> None:
        self.program = program
        self.expressions = program.rule_map()
        self.max_instructions = max_instructions
        self.instructions: list[Instruction] = []
        self.rule_entries: dict[str, int] = {}
        self.rule_nullable = {name: False for name in self.expressions}
        self._compute_nullability()

    def _compute_nullability(self) -> None:
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
        accept = self.add(Accept())
        for name, expression in self.expressions.items():
            returning = self.add(Return())
            self.rule_entries[name] = self.compile_expression(expression, returning)
        start = self.add(Call(self.program.root, accept))
        return tuple(self.instructions), self.rule_entries, start

    def compile_expression(self, expression: Expression, target: int) -> int:
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
            if expression.maximum is None and _nullable(
                expression.expression, self.rule_nullable
            ):
                raise InvalidGrammarError("Unbounded repetition of a nullable expression")
            if expression.maximum is None:
                loop = self.add(Fork(()))
                body = self.compile_expression(expression.expression, loop)
                self.instructions[loop] = Fork((body, target))
                entry = loop
                for _ in range(expression.minimum):
                    entry = self.compile_expression(expression.expression, entry)
                return entry
            entry = target
            for index in reversed(range(expression.maximum)):
                body = self.compile_expression(expression.expression, entry)
                entry = body if index < expression.minimum else self.add(Fork((body, entry)))
            return entry
        raise TypeError(type(expression))


class PushdownAutomaton:
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
        program = getattr(grammar, "program", None)
        if not isinstance(program, GrammarProgram):
            raise InvalidGrammarError(
                "The compiled grammar has no IR; compile it with GBNFCompiler"
            )
        return cls(program, **limits)

    def initial_state(self) -> AutomatonState:
        return AutomatonState(self._initial, UTF8State())

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
        if state.utf8.pending:
            return False
        return any(
            isinstance(self.instructions[configuration.pc], Accept)
            for configuration in self._closure(state.configurations)
        )

    def expected(self, state: AutomatonState) -> tuple[str, ...]:
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
