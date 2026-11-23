"""The grammar intermediate representation (IR) shared by GBNF and the custom sampler.

A grammar is a ``GrammarProgram``: named rules, each an ``Expression`` tree built
from literals, character sets, rule references, sequences, choices, repeats, and
groups. ``render_gbnf`` turns it into llama.cpp's GBNF text, and
``toolsmith.sampling.automaton`` compiles the very same tree into a pushdown
automaton. Because there is one IR, there is no second GBNF parser that could
drift from the first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class Empty:
    """Matches the empty string."""


@dataclass(frozen=True)
class Literal:
    """Matches ``text`` exactly."""

    text: str


@dataclass(frozen=True)
class CharRange:
    """An inclusive range of Unicode codepoints."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if not 0 <= self.start <= self.end <= 0x10FFFF:
            raise ValueError("Invalid Unicode codepoint range")


@dataclass(frozen=True)
class CharSet:
    """One character from (or, if ``negated``, outside) the given ranges, like ``[a-z]``/``[^"]``."""

    ranges: tuple[CharRange, ...]
    negated: bool = False

    @classmethod
    def from_characters(cls, characters: str, *, negated: bool = False) -> CharSet:
        return cls(
            tuple(CharRange(ord(character), ord(character)) for character in characters),
            negated,
        )

    @classmethod
    def single(cls, codepoint: int) -> CharSet:
        return cls((CharRange(codepoint, codepoint),))

    def matches(self, codepoint: int) -> bool:
        contained = any(item.start <= codepoint <= item.end for item in self.ranges)
        return not contained if self.negated else contained


@dataclass(frozen=True)
class RuleRef:
    """Matches whatever the named rule matches (rules may refer to each other recursively)."""

    name: str


@dataclass(frozen=True)
class Sequence:
    """Each item in order."""

    items: tuple[Expression, ...]


@dataclass(frozen=True)
class Choice:
    """Any one of the alternatives."""

    alternatives: tuple[Expression, ...]


@dataclass(frozen=True)
class Repeat:
    """``expression`` repeated ``minimum`` to ``maximum`` times (``None``: unbounded)."""

    expression: Expression
    minimum: int = 0
    maximum: int | None = None

    def __post_init__(self) -> None:
        if self.minimum < 0:
            raise ValueError("Repeat minimum must not be negative")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("Repeat maximum must not be smaller than minimum")


@dataclass(frozen=True)
class Group:
    """Parentheses in GBNF output; no effect on what matches."""

    expression: Expression


Expression: TypeAlias = Empty | Literal | CharSet | RuleRef | Sequence | Choice | Repeat | Group

EMPTY = Empty()


def sequence(*items: Expression) -> Expression:
    """Build a sequence, flattening nested sequences and collapsing zero or one item."""
    flattened = []
    for item in items:
        if isinstance(item, Sequence):
            flattened.extend(item.items)
        else:
            flattened.append(item)
    if not flattened:
        return EMPTY
    if len(flattened) == 1:
        return flattened[0]
    return Sequence(tuple(flattened))


def choice(*alternatives: Expression) -> Expression:
    """Build a choice, flattening nested choices and collapsing zero or one alternative."""
    flattened = []
    for alternative in alternatives:
        if isinstance(alternative, Choice):
            flattened.extend(alternative.alternatives)
        else:
            flattened.append(alternative)
    if not flattened:
        return EMPTY
    if len(flattened) == 1:
        return flattened[0]
    return Choice(tuple(flattened))


def optional(expression: Expression) -> Repeat:
    """Zero or one ``expression`` (GBNF ``?``)."""
    return Repeat(expression, 0, 1)


# GBNF rendering helpers: quote literals and character classes with GBNF's escapes.


def _literal(text: str) -> str:
    escaped = []
    for character in text:
        codepoint = ord(character)
        if character == '"':
            escaped.append('\\"')
        elif character == "\\":
            escaped.append("\\\\")
        elif character == "\n":
            escaped.append("\\n")
        elif character == "\r":
            escaped.append("\\r")
        elif character == "\t":
            escaped.append("\\t")
        elif codepoint < 0x20:
            escaped.append(f"\\u{codepoint:04x}")
        else:
            escaped.append(character)
    return '"' + "".join(escaped) + '"'


def _class_character(codepoint: int, *, negated: bool, final: bool) -> str:
    # Characters special inside [...] are escaped; a literal "-" is safe only last in a positive class.
    character = chr(codepoint)
    if character == "\\":
        return "\\\\"
    if character == "]":
        return "\\]"
    if character == "-" and (negated or not final):
        return "\\-"
    if character == "\t":
        return "\\t"
    if character == "\r":
        return "\\r"
    if character == "\n":
        return "\\n"
    if codepoint < 0x20 or codepoint == 0x7F:
        return f"\\x{codepoint:02X}"
    if codepoint > 0xFFFF:
        return f"\\U{codepoint:08X}"
    if codepoint > 0xFF:
        return f"\\u{codepoint:04X}"
    return character


def _charset(expression: CharSet) -> str:
    parts = []
    for index, item in enumerate(expression.ranges):
        final = index == len(expression.ranges) - 1
        start = _class_character(item.start, negated=expression.negated, final=final)
        if item.start == item.end:
            parts.append(start)
        else:
            end = _class_character(item.end, negated=expression.negated, final=final)
            parts.append(f"{start}-{end}")
    return "[^" + "".join(parts) + "]" if expression.negated else "[" + "".join(parts) + "]"


def _render(expression: Expression) -> str:
    # Sequences and choices under a repeat get parentheses, so "a | b" + "*" means "(a | b)*".
    if isinstance(expression, Empty):
        return ""
    if isinstance(expression, Literal):
        return _literal(expression.text)
    if isinstance(expression, CharSet):
        return _charset(expression)
    if isinstance(expression, RuleRef):
        return expression.name
    if isinstance(expression, Group):
        return f"({_render(expression.expression)})"
    if isinstance(expression, Sequence):
        return " ".join(_render(item) for item in expression.items)
    if isinstance(expression, Choice):
        return " | ".join(_render(item) for item in expression.alternatives)
    if isinstance(expression, Repeat):
        rendered = _render(expression.expression)
        if isinstance(expression.expression, (Sequence, Choice)):
            rendered = f"({rendered})"
        if expression.minimum == 0 and expression.maximum is None:
            suffix = "*"
        elif expression.minimum == 1 and expression.maximum is None:
            suffix = "+"
        elif expression.minimum == 0 and expression.maximum == 1:
            suffix = "?"
        elif expression.maximum == expression.minimum:
            suffix = f"{{{expression.minimum}}}"
        elif expression.maximum is None:
            suffix = f"{{{expression.minimum},}}"
        else:
            suffix = f"{{{expression.minimum},{expression.maximum}}}"
        return rendered + suffix
    raise TypeError(f"Unsupported grammar expression: {type(expression)!r}")


@dataclass(frozen=True)
class GrammarProgram:
    """A complete grammar: uniquely named rules, one of which is the ``root``."""

    rules: tuple[tuple[str, Expression], ...]
    root: str = "root"

    def __post_init__(self) -> None:
        names = [name for name, _ in self.rules]
        if len(names) != len(set(names)):
            raise ValueError("Grammar rule names must be unique")
        if self.root not in names:
            raise ValueError(f"Missing root rule: {self.root}")

    def rule_map(self) -> dict[str, Expression]:
        return dict(self.rules)

    def rule(self, name: str) -> Expression:
        for candidate, expression in self.rules:
            if candidate == name:
                return expression
        raise KeyError(name)

    def render_gbnf(self) -> str:
        """GBNF text, root rule first, the others in definition order."""
        ordered = sorted(self.rules, key=lambda item: item[0] != self.root)
        return "\n".join(f"{name} ::= {_render(expression)}" for name, expression in ordered) + "\n"


__all__ = [
    "EMPTY",
    "CharRange",
    "CharSet",
    "Choice",
    "Empty",
    "Expression",
    "GrammarProgram",
    "Group",
    "Literal",
    "Repeat",
    "RuleRef",
    "Sequence",
    "choice",
    "optional",
    "sequence",
]
