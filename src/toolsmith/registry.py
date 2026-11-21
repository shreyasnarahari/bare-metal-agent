from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TypeVar

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import ConfigDict, create_model

_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")
F = TypeVar("F", bound=Callable[..., Any])


def _validate_name(name: str) -> None:
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid tool name: {name!r}")


def _validate_parameters(parameters: Mapping[str, Any]) -> None:
    try:
        Draft202012Validator.check_schema(dict(parameters))
    except SchemaError as exc:
        raise ValueError(f"Invalid tool JSON Schema: {exc.message}") from exc
    if parameters.get("type") != "object":
        raise ValueError("Tool parameters must have JSON Schema type 'object'")


def _schema_from_function(function: Callable[..., Any], name: str) -> dict[str, Any]:
    fields = {}
    for parameter in inspect.signature(function).parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(f"Unsupported parameter kind for {parameter.name!r}")
        if parameter.annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter {parameter.name!r} needs a type annotation")
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (parameter.annotation, default)

    model = create_model(
        f"{name}_arguments",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        _validate_name(self.name)
        _validate_parameters(self.parameters)
        if self.handler is not None and not callable(self.handler):
            raise TypeError("handler must be callable")
        object.__setattr__(self, "parameters", deepcopy(self.parameters))

    @classmethod
    def from_function(
        cls,
        function: F,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> Tool:
        tool_name = name or function.__name__
        _validate_name(tool_name)
        schema = (
            _schema_from_function(function, tool_name)
            if parameters is None
            else dict(parameters)
        )
        return cls(
            name=tool_name,
            description=description or inspect.getdoc(function) or tool_name,
            parameters=schema,
            handler=function,
        )

    @classmethod
    def from_openai(cls, definition: Mapping[str, Any]) -> Tool:
        if definition.get("type") != "function":
            raise ValueError("Only function tools are supported")
        function = definition.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("A function tool needs a function object")
        name = function.get("name")
        if not isinstance(name, str):
            raise ValueError("A function tool needs a string name")
        description = function.get("description", "")
        if not isinstance(description, str):
            raise ValueError("Tool description must be a string")
        parameters = function.get(
            "parameters",
            {"type": "object", "properties": {}, "additionalProperties": False},
        )
        if not isinstance(parameters, Mapping):
            raise ValueError("Tool parameters must be an object")
        return cls(name, description, dict(parameters))

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": deepcopy(self.parameters),
            },
        }


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def function(
        self,
        function: F | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> F | Callable[[F], F]:
        def decorate(target: F) -> F:
            self.register(
                Tool.from_function(
                    target,
                    name=name,
                    description=description,
                    parameters=parameters,
                )
            )
            return target

        return decorate(function) if function is not None else decorate

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def to_openai(self) -> list[dict[str, Any]]:
        return [tool.to_openai() for tool in self._tools.values()]

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
