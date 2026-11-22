"""``SteadyhandRegistry``: Toolsmith tools plus a policy and semantic-check side table.

Tools are still ``toolsmith.registry.Tool`` objects built through
``ToolRegistry.function``, so ``schemas()`` is exactly what the model sees.
Handlers take the episode store first; the model-facing schema covers only the
parameters after it. Policies are checked against each other at registration,
and ``freeze()`` makes the registry read-only so workers can share it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator

from toolsmith.evaluation.models import canonical_hash
from toolsmith.registry import Tool, ToolRegistry

from .errors import FieldProblem
from .policy import Effect, PolicyError, ToolPolicy

if TYPE_CHECKING:
    from .store import Store

SemanticCheck = Callable[["Store", Mapping[str, Any]], list[FieldProblem]]

# Tool names a policy may point at before the tool exists. Part 6 registers these.
RESERVED_TOOL_NAMES = frozenset({"refund_via_ledger"})


def arguments_schema(handler: Callable[..., Any]) -> dict[str, Any]:
    """The JSON Schema of every parameter after ``store``, built by ``Tool.from_function``."""
    signature = inspect.signature(handler, eval_str=True)
    parameters = list(signature.parameters.values())
    if not parameters or parameters[0].name != "store":
        raise TypeError(f"{handler.__name__} must take the store as its first parameter")

    def arguments_only(**arguments: Any) -> Any:
        raise NotImplementedError

    arguments_only.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=parameters[1:],
        return_annotation=inspect.Signature.empty,
    )
    return Tool.from_function(arguments_only, name=handler.__name__).parameters


class SteadyhandRegistry:
    def __init__(self, *, reserved_names: Iterable[str] = RESERVED_TOOL_NAMES) -> None:
        self._tools = ToolRegistry()
        self._policies: dict[str, ToolPolicy] = {}
        self._semantics: dict[str, SemanticCheck] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        self._result_schemas: dict[str, dict[str, Any]] = {}
        self._reserved = frozenset(reserved_names)
        self._frozen = False

    def register(
        self,
        function: Callable[..., Any],
        policy: ToolPolicy,
        semantic: SemanticCheck | None = None,
        result_schema: Mapping[str, Any] | None = None,
    ) -> Tool:
        """Register a handler. ``result_schema`` describes its return value; the model never sees it."""
        if self._frozen:
            raise RuntimeError("The registry is frozen; register every tool before sharing it")
        name = function.__name__
        if name in self._policies:
            raise ValueError(f"Tool {name!r} is already registered")
        self._check_policy(name, policy)
        if semantic is not None and not callable(semantic):
            raise TypeError("semantic must be callable")
        self._tools.function(function, parameters=arguments_schema(function))
        tool = self._tools.get(name)
        self._policies[name] = policy
        self._validators[name] = Draft202012Validator(tool.parameters)
        if semantic is not None:
            self._semantics[name] = semantic
        if result_schema is not None:
            Draft202012Validator.check_schema(dict(result_schema))
            self._result_schemas[name] = dict(result_schema)
        return tool

    def freeze(self) -> SteadyhandRegistry:
        self._frozen = True
        return self

    @property
    def frozen(self) -> bool:
        return self._frozen

    def get(self, name: str) -> Tool:
        return self._tools.get(name)

    def policy(self, name: str) -> ToolPolicy:
        try:
            return self._policies[name]
        except KeyError:
            raise KeyError(f"Unknown tool: {name}") from None

    def semantic(self, name: str) -> SemanticCheck | None:
        return self._semantics.get(name)

    def result_schema(self, name: str) -> dict[str, Any] | None:
        return self._result_schemas.get(name)

    def validator(self, name: str) -> Draft202012Validator:
        try:
            return self._validators[name]
        except KeyError:
            raise KeyError(f"Unknown tool: {name}") from None

    def schemas(self) -> list[dict[str, Any]]:
        """The OpenAI tool list sent to the model. A fresh copy on every call."""
        return self._tools.to_openai()

    def schemas_sha256(self) -> str:
        return canonical_hash(self.schemas())

    def policies_sha256(self) -> str:
        return canonical_hash(
            [[name, self._policies[name].as_dict()] for name in sorted(self._policies)]
        )

    def names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._policies

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def _check_policy(self, name: str, policy: ToolPolicy) -> None:
        if policy.verify is not None:
            if policy.verify == name:
                raise PolicyError(f"{name}: verify cannot point at the tool itself")
            if policy.verify not in self._policies:
                raise PolicyError(
                    f"{name}: verify tool {policy.verify!r} is not registered "
                    "(register read tools before the writes they verify)"
                )
            if self._policies[policy.verify].effect is not Effect.READ:
                raise PolicyError(f"{name}: verify tool {policy.verify!r} must be a read")
        for alternative in policy.alternatives:
            if alternative == name:
                raise PolicyError(f"{name}: a tool cannot be its own alternative")
            if alternative not in self._policies and alternative not in self._reserved:
                raise PolicyError(
                    f"{name}: alternative {alternative!r} is neither registered nor reserved"
                )


__all__ = ["RESERVED_TOOL_NAMES", "SemanticCheck", "SteadyhandRegistry", "arguments_schema"]
