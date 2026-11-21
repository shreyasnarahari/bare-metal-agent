from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .protocols import ToolProtocol
from .registry import Tool
from .sampling.ir import (
    EMPTY,
    CharRange,
    CharSet,
    Empty,
    Expression,
    GrammarProgram,
    Group,
    Literal,
    Repeat,
    RuleRef,
    choice,
    optional,
    sequence,
)

_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_COMMON_KEYWORDS = {
    "$schema",
    "default",
    "description",
    "enum",
    "examples",
    "title",
    "type",
}
_OBJECT_KEYWORDS = _COMMON_KEYWORDS | {"additionalProperties", "properties", "required"}
_ARRAY_KEYWORDS = _COMMON_KEYWORDS | {"items"}
_PROTOCOL_MARKERS = ("<tool_call>", "</tool_call>", "<think>", "</think>")
_WS = RuleRef("json-ws")


class SchemaNotSupportedError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        pointer: str = "",
        keyword: str | None = None,
    ) -> None:
        self.pointer = pointer or "/"
        self.keyword = keyword
        location = f" at {self.pointer}" if self.pointer else ""
        super().__init__(f"{message}{location}")


class GrammarConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ToolChoice:
    mode: str
    name: str | None = None

    def as_dict(self) -> dict[str, str]:
        value = {"mode": self.mode}
        if self.name is not None:
            value["name"] = self.name
        return value


ToolChoiceValue: TypeAlias = str | Mapping[str, Any] | ToolChoice | None


@dataclass(frozen=True)
class CompiledGrammar:
    text: str
    mode: str
    tool_names: tuple[str, ...] = ()
    program: GrammarProgram | None = field(default=None, compare=False, repr=False)
    protocol: str = "qwen3"
    limitations: tuple[str, ...] = ()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def rule_count(self) -> int:
        return sum(" ::= " in line for line in self.text.splitlines())


def resolve_tool_choice(
    tools: Sequence[Tool],
    value: ToolChoiceValue,
) -> ToolChoice:
    names = {tool.name for tool in tools}
    if isinstance(value, ToolChoice):
        choice = value
    elif value is None:
        choice = ToolChoice("auto")
    elif isinstance(value, str):
        if value not in {"auto", "none", "required"}:
            raise GrammarConfigurationError(f"Unsupported tool_choice: {value!r}")
        choice = ToolChoice(value)
    elif isinstance(value, Mapping):
        if set(value) != {"type", "function"} or value.get("type") != "function":
            raise GrammarConfigurationError(
                "A named tool_choice must contain only type='function' and function"
            )
        function = value.get("function")
        if not isinstance(function, Mapping) or set(function) != {"name"}:
            raise GrammarConfigurationError(
                "A named tool_choice function must contain only a name"
            )
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise GrammarConfigurationError("A named tool_choice needs a non-empty name")
        choice = ToolChoice("named", name)
    else:
        raise GrammarConfigurationError("tool_choice must be a string, function choice, or null")

    if not tools and choice.mode in {"required", "named"}:
        raise GrammarConfigurationError(
            f"tool_choice={choice.mode!r} requires at least one registered tool"
        )
    if choice.mode == "named" and choice.name not in names:
        raise GrammarConfigurationError(f"Unknown tool in tool_choice: {choice.name!r}")
    if choice.mode == "named" and choice.name is None:
        raise GrammarConfigurationError("A named tool choice needs a tool name")
    if choice.mode not in {"auto", "none", "required", "named"}:
        raise GrammarConfigurationError(f"Unsupported normalized tool choice: {choice.mode!r}")
    return choice


class _RuleBuilder:
    def __init__(self, max_rules: int) -> None:
        self.max_rules = max_rules
        self.rules: dict[str, Expression] = {}
        self.next_id = 0
        self.add(
            "json-ws",
            Repeat(CharSet.from_characters(" \t\r\n"), 0, 4),
        )

    def name(self, prefix: str) -> str:
        name = f"{prefix}-{self.next_id}"
        self.next_id += 1
        return name

    def add(self, name: str, expression: Expression) -> str:
        if name not in self.rules and len(self.rules) >= self.max_rules:
            raise GrammarConfigurationError(
                f"The schema exceeds the configured limit of {self.max_rules} GBNF rules"
            )
        self.rules[name] = expression
        return name

    def primitive(self, kind: str) -> str:
        name = f"json-{kind}"
        if name in self.rules:
            return name
        if kind == "string":
            self.add(
                "json-char",
                choice(
                    CharSet(
                        (
                            CharRange(ord('"'), ord('"')),
                            CharRange(ord("\\"), ord("\\")),
                            CharRange(0x00, 0x1F),
                        ),
                        negated=True,
                    ),
                    sequence(
                        CharSet.from_characters("\\"),
                        Group(
                            choice(
                                CharSet.from_characters('"\\/bfnrt'),
                                sequence(
                                    Literal("u"),
                                    Repeat(
                                        CharSet(
                                            (
                                                CharRange(ord("0"), ord("9")),
                                                CharRange(ord("a"), ord("f")),
                                                CharRange(ord("A"), ord("F")),
                                            )
                                        ),
                                        4,
                                        4,
                                    ),
                                ),
                            )
                        ),
                    ),
                ),
            )
            self.add(
                name,
                sequence(
                    Literal('"'),
                    Repeat(RuleRef("json-char")),
                    Literal('"'),
                ),
            )
        elif kind in {"integer", "number"}:
            digits = CharSet((CharRange(ord("0"), ord("9")),))
            integral = Group(
                choice(
                    CharSet.from_characters("0"),
                    sequence(
                        CharSet((CharRange(ord("1"), ord("9")),)),
                        Repeat(digits),
                    ),
                )
            )
            parts = [optional(Literal("-")), integral]
            if kind == "number":
                parts.extend(
                    [
                        optional(Group(sequence(Literal("."), Repeat(digits, 1)))),
                        optional(
                            Group(
                                sequence(
                                    CharSet.from_characters("eE"),
                                    optional(CharSet.from_characters("+-")),
                                    Repeat(digits, 1),
                                )
                            )
                        ),
                    ]
                )
            self.add(name, sequence(*parts))
        elif kind == "boolean":
            self.add(name, choice(Literal("true"), Literal("false")))
        elif kind == "null":
            self.add(name, Literal("null"))
        else:
            raise AssertionError(kind)
        return name

    def program(self) -> GrammarProgram:
        return GrammarProgram(tuple(self.rules.items()))

    def render(self) -> str:
        return self.program().render_gbnf()


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _json_literal(value: Any) -> Literal:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return Literal(encoded)


class GBNFCompiler:
    def __init__(
        self,
        *,
        max_depth: int = 32,
        max_rules: int = 512,
        max_tools: int = 64,
        protocol: ToolProtocol | str = ToolProtocol.QWEN3,
    ) -> None:
        if min(max_depth, max_rules, max_tools) < 1:
            raise ValueError("Grammar limits must be positive")
        try:
            self.protocol = ToolProtocol(protocol)
        except ValueError as exc:
            raise ValueError(f"Unsupported tool protocol: {protocol!r}") from exc
        self.max_depth = max_depth
        self.max_rules = max_rules
        self.max_tools = max_tools

    def compile_schema(self, schema: Mapping[str, Any]) -> CompiledGrammar:
        self._check_schema(schema)
        builder = _RuleBuilder(self.max_rules)
        value_rule = self._compile_node(builder, schema, "", 0)
        builder.add("root", sequence(_WS, RuleRef(value_rule), _WS))
        return CompiledGrammar(
            builder.render(),
            "schema",
            program=builder.program(),
            protocol=self.protocol.value,
        )

    def compile_tools(
        self,
        tools: Sequence[Tool],
        tool_choice: ToolChoiceValue = "auto",
    ) -> CompiledGrammar | None:
        tools = tuple(tools)
        if len(tools) > self.max_tools:
            raise GrammarConfigurationError(
                f"The request exceeds the configured limit of {self.max_tools} tools"
            )
        normalized = resolve_tool_choice(tools, tool_choice)
        if not tools:
            return None

        builder = _RuleBuilder(self.max_rules)
        selected = (
            tuple(tool for tool in tools if tool.name == normalized.name)
            if normalized.mode == "named"
            else tools
        )
        call_rules: list[str] = []
        if normalized.mode != "none":
            for tool in selected:
                call_rules.append(self._compile_tool(builder, tool))

        if normalized.mode == "none":
            self._add_plain_text(builder)
            builder.add("root", RuleRef("plain-text"))
        elif normalized.mode == "named":
            builder.add("root", RuleRef(call_rules[0]))
        else:
            alternatives: Expression = choice(*(RuleRef(name) for name in call_rules))
            if len(call_rules) > 1:
                alternatives = Group(alternatives)
            builder.add(
                "tool-calls",
                sequence(
                    alternatives,
                    Repeat(Group(sequence(_WS, alternatives))),
                ),
            )
            if normalized.mode == "auto":
                self._add_plain_text(builder)
                builder.add("root", choice(RuleRef("plain-text"), RuleRef("tool-calls")))
            else:
                builder.add("root", RuleRef("tool-calls"))
        limitations = (
            ("Qwen3.5 raw string parameters cannot contain reserved tool markup",)
            if self.protocol is ToolProtocol.QWEN35
            else ()
        )
        return CompiledGrammar(
            builder.render(),
            normalized.mode,
            tuple(tool.name for tool in selected),
            builder.program(),
            self.protocol.value,
            limitations,
        )

    def _check_schema(self, schema: Mapping[str, Any]) -> None:
        if not isinstance(schema, Mapping):
            raise SchemaNotSupportedError("A schema must be a JSON object")
        try:
            json.dumps(schema, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise SchemaNotSupportedError(
                "The schema must contain only finite, JSON-compatible values"
            ) from exc
        try:
            Draft202012Validator.check_schema(dict(schema))
        except SchemaError as exc:
            pointer = "/" + "/".join(str(part) for part in exc.absolute_schema_path)
            raise SchemaNotSupportedError(
                f"Invalid Draft 2020-12 schema: {exc.message}",
                pointer=pointer,
            ) from exc
        dialect = schema.get("$schema", _DRAFT_2020_12)
        if dialect != _DRAFT_2020_12:
            raise SchemaNotSupportedError(
                "Only the Draft 2020-12 dialect is supported",
                pointer="/$schema",
                keyword="$schema",
            )

    def _validate_node(
        self,
        schema: Mapping[str, Any],
        pointer: str,
        depth: int,
    ) -> None:
        if depth > self.max_depth:
            raise SchemaNotSupportedError(
                f"The schema exceeds the configured depth of {self.max_depth}",
                pointer=pointer,
            )
        if not isinstance(schema, Mapping):
            raise SchemaNotSupportedError("Boolean schemas are not supported", pointer=pointer)
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            raise SchemaNotSupportedError(
                "Type unions are not supported",
                pointer=pointer + "/type",
                keyword="type",
            )
        if schema_type is not None and schema_type not in {
            "object",
            "array",
            "string",
            "integer",
            "number",
            "boolean",
            "null",
        }:
            raise SchemaNotSupportedError(
                f"Unsupported schema type: {schema_type!r}",
                pointer=pointer + "/type",
                keyword="type",
            )

        allowed = _COMMON_KEYWORDS
        if "enum" not in schema:
            if schema_type == "object":
                allowed = _OBJECT_KEYWORDS
            elif schema_type == "array":
                allowed = _ARRAY_KEYWORDS
        for keyword in schema:
            if keyword not in allowed:
                raise SchemaNotSupportedError(
                    f"Unsupported JSON Schema keyword: {keyword!r}",
                    pointer=pointer + "/" + _pointer_token(keyword),
                    keyword=keyword,
                )
        if schema_type is None and "enum" not in schema:
            raise SchemaNotSupportedError(
                "A supported type or enum is required",
                pointer=pointer,
                keyword="type",
            )

    def _compile_node(
        self,
        builder: _RuleBuilder,
        schema: Mapping[str, Any],
        pointer: str,
        depth: int,
    ) -> str:
        self._validate_node(schema, pointer, depth)
        schema_type = schema.get("type")
        if "enum" in schema:
            return self._compile_enum(builder, schema, pointer)
        if schema_type == "object":
            return self._compile_object(builder, schema, pointer, depth)
        if schema_type == "array":
            return self._compile_array(builder, schema, pointer, depth)
        return builder.primitive(schema_type)

    def _compile_enum(
        self,
        builder: _RuleBuilder,
        schema: Mapping[str, Any],
        pointer: str,
    ) -> str:
        values = schema["enum"]
        schema_type = schema.get("type")
        alternatives = []
        for index, value in enumerate(values):
            if value is not None and not isinstance(value, (str, bool, int, float)):
                raise SchemaNotSupportedError(
                    "Only scalar enum values are supported",
                    pointer=f"{pointer}/enum/{index}",
                    keyword="enum",
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise SchemaNotSupportedError(
                    "Enum numbers must be finite",
                    pointer=f"{pointer}/enum/{index}",
                    keyword="enum",
                )
            if schema_type is not None and not Draft202012Validator(
                {"type": schema_type}
            ).is_valid(value):
                raise SchemaNotSupportedError(
                    f"Enum value does not satisfy type {schema_type!r}",
                    pointer=f"{pointer}/enum/{index}",
                    keyword="enum",
                )
            alternatives.append(_json_literal(value))
        expression = choice(*alternatives)
        if len(alternatives) > 1:
            expression = Group(expression)
        return builder.add(builder.name("enum"), expression)

    def _compile_object(
        self,
        builder: _RuleBuilder,
        schema: Mapping[str, Any],
        pointer: str,
        depth: int,
    ) -> str:
        if schema.get("additionalProperties") is not False:
            raise SchemaNotSupportedError(
                "Objects must explicitly set additionalProperties to false",
                pointer=pointer + "/additionalProperties",
                keyword="additionalProperties",
            )
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise SchemaNotSupportedError(
                "properties must be an object",
                pointer=pointer + "/properties",
                keyword="properties",
            )
        required = set(schema.get("required", []))
        missing = required - set(properties)
        if missing:
            raise SchemaNotSupportedError(
                f"Required properties are not declared: {sorted(missing)!r}",
                pointer=pointer + "/required",
                keyword="required",
            )

        name = builder.name("object")
        key_value_rules: dict[str, str] = {}
        for property_name, property_schema in properties.items():
            property_pointer = pointer + "/properties/" + _pointer_token(property_name)
            value_rule = self._compile_node(
                builder,
                property_schema,
                property_pointer,
                depth + 1,
            )
            key_value_rules[property_name] = builder.add(
                builder.name("property"),
                sequence(
                    _json_literal(property_name),
                    _WS,
                    Literal(":"),
                    _WS,
                    RuleRef(value_rule),
                ),
            )

        required_names = [name for name in properties if name in required]
        optional_names = [name for name in properties if name not in required]
        required_parts: list[Expression] = []
        for property_name in required_names:
            if required_parts:
                required_parts.extend([Literal(","), _WS])
            required_parts.append(RuleRef(key_value_rules[property_name]))
        members: Expression = sequence(*required_parts)
        if optional_names:
            alternatives = []
            for start in range(len(optional_names)):
                alternatives.append(
                    self._optional_sequence(
                        builder,
                        name,
                        optional_names[start:],
                        key_value_rules,
                        first_is_optional=False,
                    )
                )
            optional_expression = choice(*alternatives)
            if required_names:
                members = sequence(
                    members,
                    Group(
                        optional(
                            Group(
                                sequence(
                                    Literal(","),
                                    _WS,
                                    Group(optional_expression),
                                )
                            )
                        )
                    ),
                )
            else:
                members = optional(Group(optional_expression))
        expression = (
            sequence(Literal("{"), _WS, members, _WS, Literal("}"))
            if not isinstance(members, Empty)
            else sequence(Literal("{"), _WS, Literal("}"))
        )
        return builder.add(name, expression)

    def _optional_sequence(
        self,
        builder: _RuleBuilder,
        owner: str,
        names: Sequence[str],
        key_value_rules: Mapping[str, str],
        *,
        first_is_optional: bool,
    ) -> Expression:
        current, *remaining = names
        current_reference = Group(
            sequence(Literal(","), _WS, RuleRef(key_value_rules[current]))
        )
        expression: Expression = (
            optional(current_reference)
            if first_is_optional
            else RuleRef(key_value_rules[current])
        )
        if remaining:
            rest_name = builder.name(f"{owner}-optional")
            rest_expression = self._optional_sequence(
                builder,
                owner,
                remaining,
                key_value_rules,
                first_is_optional=True,
            )
            builder.add(rest_name, rest_expression)
            expression = sequence(expression, RuleRef(rest_name))
        return expression

    def _compile_array(
        self,
        builder: _RuleBuilder,
        schema: Mapping[str, Any],
        pointer: str,
        depth: int,
    ) -> str:
        if "items" not in schema:
            raise SchemaNotSupportedError(
                "Arrays require an inline items schema",
                pointer=pointer + "/items",
                keyword="items",
            )
        name = builder.name("array")
        item_rule = self._compile_node(
            builder,
            schema["items"],
            pointer + "/items",
            depth + 1,
        )
        item = RuleRef(item_rule)
        return builder.add(
            name,
            sequence(
                Literal("["),
                _WS,
                optional(
                    Group(
                        sequence(
                            item,
                            Repeat(Group(sequence(_WS, Literal(","), _WS, item))),
                        )
                    )
                ),
                _WS,
                Literal("]"),
            ),
        )

    def _compile_xml_enum(
        self,
        builder: _RuleBuilder,
        schema: Mapping[str, Any],
        pointer: str,
    ) -> str:
        self._compile_enum(builder, schema, pointer)
        markers = (
            "<tool_call>",
            "</tool_call>",
            "<parameter=",
            "</parameter>",
            "<function=",
            "</function>",
        )
        alternatives = []
        encoded_values: set[str] = set()
        for index, value in enumerate(schema["enum"]):
            encoded = value if isinstance(value, str) else _json_literal(value).text
            if any(marker in encoded for marker in markers):
                raise SchemaNotSupportedError(
                    "Enum value contains reserved Qwen3.5 parameter markup",
                    pointer=f"{pointer}/enum/{index}",
                    keyword="enum",
                )
            if encoded in encoded_values:
                raise SchemaNotSupportedError(
                    "Enum values have ambiguous Qwen3.5 parameter encodings",
                    pointer=f"{pointer}/enum/{index}",
                    keyword="enum",
                )
            encoded_values.add(encoded)
            alternatives.append(Literal(encoded))
        expression = choice(*alternatives)
        if len(alternatives) > 1:
            expression = Group(expression)
        return builder.add(builder.name("xml-enum"), expression)

    def _add_xml_text(self, builder: _RuleBuilder) -> str:
        if "xml-text" in builder.rules:
            return "xml-text"
        markers = (
            "<tool_call>",
            "</tool_call>",
            "<parameter=",
            "</parameter>",
            "<function=",
            "</function>",
        )
        prefixes = sorted(
            {
                marker[:length]
                for marker in markers
                for length in range(len(marker))
            },
            key=lambda prefix: (len(prefix), prefix),
        )
        state_names = {
            prefix: builder.name("xml-text-state") for prefix in prefixes
        }

        def target_for(text: str) -> str:
            return max(
                (prefix for prefix in prefixes if text.endswith(prefix)),
                key=len,
            )

        def transitions(prefix: str) -> list[Expression]:
            expected = {
                marker[len(prefix)]
                for marker in markers
                if marker.startswith(prefix) and len(marker) > len(prefix)
            }
            result = []
            for character in sorted(expected | {"<"}):
                following = prefix + character
                if following in markers:
                    continue
                result.append(
                    sequence(
                        Literal(character),
                        RuleRef(state_names[target_for(following)]),
                    )
                )
            default_target = target_for(prefix + "\ue000")
            default_class = CharSet.from_characters(
                "".join(sorted(expected | {"<"})),
                negated=True,
            )
            result.append(
                sequence(default_class, RuleRef(state_names[default_target]))
            )
            return result

        for prefix in prefixes:
            builder.add(
                state_names[prefix],
                choice(EMPTY, *transitions(prefix)),
            )
        builder.add("xml-text", choice(EMPTY, *transitions("")))
        return "xml-text"

    def _compile_xml_object(
        self,
        builder: _RuleBuilder,
        schema: Mapping[str, Any],
        pointer: str,
        depth: int,
    ) -> Expression:
        self._validate_node(schema, pointer, depth)
        if schema.get("additionalProperties") is not False:
            raise SchemaNotSupportedError(
                "Objects must explicitly set additionalProperties to false",
                pointer=pointer + "/additionalProperties",
                keyword="additionalProperties",
            )
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise SchemaNotSupportedError(
                "properties must be an object",
                pointer=pointer + "/properties",
                keyword="properties",
            )
        required = set(schema.get("required", []))
        missing = required - set(properties)
        if missing:
            raise SchemaNotSupportedError(
                f"Required properties are not declared: {sorted(missing)!r}",
                pointer=pointer + "/required",
                keyword="required",
            )

        name = builder.name("xml-object")
        key_value_rules: dict[str, str] = {}
        for property_name, property_schema in properties.items():
            property_pointer = pointer + "/properties/" + _pointer_token(property_name)
            if (
                not property_name
                or any(character in property_name for character in "<>\r\n")
            ):
                raise SchemaNotSupportedError(
                    "Property name cannot be represented as a Qwen3.5 parameter",
                    pointer=property_pointer,
                    keyword="properties",
                )
            json_rule = self._compile_node(
                builder,
                property_schema,
                property_pointer,
                depth + 1,
            )
            if "enum" in property_schema:
                value_rule = self._compile_xml_enum(
                    builder, property_schema, property_pointer
                )
            elif property_schema.get("type") == "string":
                value_rule = self._add_xml_text(builder)
            else:
                value_rule = json_rule
            key_value_rules[property_name] = builder.add(
                builder.name("xml-parameter"),
                sequence(
                    Literal(f"<parameter={property_name}>"),
                    Literal("\n"),
                    RuleRef(value_rule),
                    Literal("\n</parameter>"),
                ),
            )

        required_names = [name for name in properties if name in required]
        optional_names = [name for name in properties if name not in required]
        required_parts: list[Expression] = []
        for property_name in required_names:
            if required_parts:
                required_parts.append(Literal("\n"))
            required_parts.append(RuleRef(key_value_rules[property_name]))
        members: Expression = sequence(*required_parts)
        if optional_names:
            alternatives = []
            for start in range(len(optional_names)):
                alternatives.append(
                    self._xml_optional_sequence(
                        builder,
                        name,
                        optional_names[start:],
                        key_value_rules,
                        first_is_optional=False,
                    )
                )
            optional_expression = choice(*alternatives)
            if required_names:
                members = sequence(
                    members,
                    Group(
                        optional(
                            Group(
                                sequence(
                                    Literal("\n"),
                                    Group(optional_expression),
                                )
                            )
                        )
                    ),
                )
            else:
                members = optional(Group(optional_expression))
        return members

    def _xml_optional_sequence(
        self,
        builder: _RuleBuilder,
        owner: str,
        names: Sequence[str],
        key_value_rules: Mapping[str, str],
        *,
        first_is_optional: bool,
    ) -> Expression:
        current, *remaining = names
        current_reference = Group(
            sequence(Literal("\n"), RuleRef(key_value_rules[current]))
        )
        expression: Expression = (
            optional(current_reference)
            if first_is_optional
            else RuleRef(key_value_rules[current])
        )
        if remaining:
            rest_name = builder.name(f"{owner}-optional")
            rest_expression = self._xml_optional_sequence(
                builder,
                owner,
                remaining,
                key_value_rules,
                first_is_optional=True,
            )
            builder.add(rest_name, rest_expression)
            expression = sequence(expression, RuleRef(rest_name))
        return expression

    def _compile_tool(self, builder: _RuleBuilder, tool: Tool) -> str:
        self._check_schema(tool.parameters)
        if tool.parameters.get("type") != "object":
            raise SchemaNotSupportedError(
                "Tool parameters must have type 'object'",
                pointer="/type",
                keyword="type",
            )
        if self.protocol is ToolProtocol.QWEN35:
            arguments = self._compile_xml_object(builder, tool.parameters, "", 0)
            return builder.add(
                builder.name("tool"),
                sequence(
                    Literal(f"<tool_call>\n<function={tool.name}>\n"),
                    arguments,
                    EMPTY if isinstance(arguments, Empty) else Literal("\n"),
                    Literal("</function>\n</tool_call>"),
                ),
            )
        arguments = self._compile_node(builder, tool.parameters, "", 0)
        return builder.add(
            builder.name("tool"),
            sequence(
                Literal("<tool_call>"),
                _WS,
                Literal("{"),
                _WS,
                Literal('"name"'),
                _WS,
                Literal(":"),
                _WS,
                _json_literal(tool.name),
                _WS,
                Literal(","),
                _WS,
                Literal('"arguments"'),
                _WS,
                Literal(":"),
                _WS,
                RuleRef(arguments),
                _WS,
                Literal("}"),
                _WS,
                Literal("</tool_call>"),
            ),
        )

    def _add_plain_text(self, builder: _RuleBuilder) -> None:
        prefixes = sorted(
            {
                marker[:length]
                for marker in _PROTOCOL_MARKERS
                for length in range(len(marker))
            },
            key=lambda prefix: (len(prefix), prefix),
        )
        state_names = {
            prefix: builder.name("text-state") for prefix in prefixes
        }

        def target_for(text: str) -> str:
            return max(
                (prefix for prefix in prefixes if text.endswith(prefix)),
                key=len,
            )

        def transitions(prefix: str) -> list[Expression]:
            expected = {
                marker[len(prefix)]
                for marker in _PROTOCOL_MARKERS
                if marker.startswith(prefix) and len(marker) > len(prefix)
            }
            result = []
            for character in sorted(expected | {"<"}):
                following = prefix + character
                if following in _PROTOCOL_MARKERS:
                    continue
                result.append(
                    sequence(
                        Literal(character),
                        RuleRef(state_names[target_for(following)]),
                    )
                )
            default_target = target_for(prefix + "\ue000")
            default_class = CharSet.from_characters(
                "".join(sorted(expected | {"<"})),
                negated=True,
            )
            result.append(
                sequence(default_class, RuleRef(state_names[default_target]))
            )
            return result

        for prefix in prefixes:
            builder.add(
                state_names[prefix],
                choice(EMPTY, *transitions(prefix)),
            )
        builder.add("plain-text", choice(*transitions("")))


__all__ = [
    "CompiledGrammar",
    "GBNFCompiler",
    "GrammarConfigurationError",
    "SchemaNotSupportedError",
    "ToolChoice",
    "ToolChoiceValue",
    "resolve_tool_choice",
]
