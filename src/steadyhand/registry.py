"""``SteadyhandRegistry``: Toolsmith tools plus policy, security, and semantic-check side tables.

Tools are still ``toolsmith.registry.Tool`` objects built through
``ToolRegistry.function``, so ``schemas()`` is exactly what the model sees.
Handlers take the episode store first; the model-facing schema covers only the
parameters after it. Policies are checked against each other at registration,
and ``freeze()`` makes the registry read-only so workers can share it.

Bulkhead security metadata (``SecuritySpec``) is a separate side table, never
part of a model-facing schema. An ``Effect.EXEC`` tool must be registered with
a spec that has a sandbox. With Bulkhead enabled, the policy gate denies any
tool that has no spec at all; a missing spec is never treated as full access.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator

from bulkhead.backends.base import ProgramSpec
from bulkhead.manifest import SecuritySpec
from toolsmith.evaluation.models import canonical_hash
from toolsmith.registry import Tool, ToolRegistry

from .errors import FieldProblem
from .policy import Effect, PolicyError, ToolPolicy

if TYPE_CHECKING:
    from .store import Store

# A tool's extra argument check against the store (ids exist, amounts allowed), beyond its schema.
SemanticCheck = Callable[["Store", Mapping[str, Any]], list[FieldProblem]]

# Tool names a policy may point at that the model-facing registry does not hold.
# refund_via_ledger lives in the switch strategy's alternatives registry (part 6).
RESERVED_TOOL_NAMES = frozenset({"refund_via_ledger"})
# Server-side parameters the executor supplies. Never in the model-facing schema,
# so a model cannot pass its own idempotency key (the closed schema rejects it).
SERVER_PARAMETERS = frozenset({"idempotency_key"})


def arguments_schema(handler: Callable[..., Any]) -> dict[str, Any]:
    """The JSON Schema of every parameter after ``store``, built by ``Tool.from_function``."""
    signature = inspect.signature(handler, eval_str=True)
    parameters = list(signature.parameters.values())
    if not parameters or parameters[0].name != "store":
        raise TypeError(f"{handler.__name__} must take the store as its first parameter")

    # A stand-in whose signature is the handler's minus store and server parameters; only its
    # schema is used (the description comes from the real handler's docstring at registration).
    def arguments_only(**arguments: Any) -> Any:
        raise NotImplementedError

    arguments_only.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[parameter for parameter in parameters[1:] if parameter.name not in SERVER_PARAMETERS],
        return_annotation=inspect.Signature.empty,
    )
    return Tool.from_function(arguments_only, name=handler.__name__).parameters


class SteadyhandRegistry:
    """Model-facing tools plus their policies, semantic checks, argument validators, and result schemas."""

    def __init__(self, *, reserved_names: Iterable[str] = RESERVED_TOOL_NAMES) -> None:
        self._tools = ToolRegistry()
        self._policies: dict[str, ToolPolicy] = {}
        self._semantics: dict[str, SemanticCheck] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        self._result_schemas: dict[str, dict[str, Any]] = {}
        self._security: dict[str, SecuritySpec | None] = {}
        self._programs: dict[str, ProgramSpec] = {}
        self._reserved = frozenset(reserved_names)
        self._frozen = False

    def register(
        self,
        function: Callable[..., Any],
        policy: ToolPolicy,
        semantic: SemanticCheck | None = None,
        result_schema: Mapping[str, Any] | None = None,
        security: SecuritySpec | None = None,
        program: ProgramSpec | None = None,
    ) -> Tool:
        """Register a handler. ``result_schema`` describes its return value; the model never sees it.

        ``security`` is the tool's Bulkhead metadata. Exec tools must have one with a sandbox.
        ``program`` is an exec tool's dispatch descriptor: the program kind it runs and the
        required string argument holding the text. Only exec tools take one.
        """
        if self._frozen:
            raise RuntimeError("The registry is frozen; register every tool before sharing it")
        name = function.__name__
        if name in self._policies:
            raise ValueError(f"Tool {name!r} is already registered")
        self._check_policy(name, policy)
        if security is not None and not isinstance(security, SecuritySpec):
            raise TypeError("security must be a bulkhead SecuritySpec")
        if policy.effect is Effect.EXEC and (security is None or not security.sandboxed):
            raise PolicyError(f"{name}: an exec tool must be registered with a SecuritySpec that has a sandbox")
        if semantic is not None and not callable(semantic):
            raise TypeError("semantic must be callable")
        if program is not None:
            if not isinstance(program, ProgramSpec):
                raise TypeError("program must be a bulkhead ProgramSpec")
            if policy.effect is not Effect.EXEC:
                raise PolicyError(f"{name}: only an exec tool has a program descriptor")
        parameters = arguments_schema(function)
        if program is not None:
            prop = parameters.get("properties", {}).get(program.argument, {})
            if prop.get("type") != "string" or program.argument not in parameters.get("required", ()):
                raise PolicyError(f"{name}: the program argument {program.argument!r} must be a required string")
        self._tools.function(function, parameters=parameters)
        tool = self._tools.get(name)
        self._policies[name] = policy
        self._security[name] = security
        if program is not None:
            self._programs[name] = program
        self._validators[name] = Draft202012Validator(tool.parameters)
        if semantic is not None:
            self._semantics[name] = semantic
        if result_schema is not None:
            Draft202012Validator.check_schema(dict(result_schema))
            self._result_schemas[name] = dict(result_schema)
        return tool

    def freeze(self) -> SteadyhandRegistry:
        """Make the registry read-only, so it can be shared across threads and processes."""
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

    def program(self, name: str) -> ProgramSpec | None:
        """An exec tool's dispatch descriptor, or None."""
        if name not in self._policies:
            raise KeyError(f"Unknown tool: {name}")
        return self._programs.get(name)

    def security(self, name: str) -> SecuritySpec | None:
        """The tool's Bulkhead security spec, or None when it was registered without one."""
        try:
            return self._security[name]
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
        """Hash of exactly what the model sees; part of every episode manifest."""
        return canonical_hash(self.schemas())

    def policies_sha256(self) -> str:
        """Hash of every policy, by tool name; a policy change gives every episode a new id."""
        return canonical_hash(
            [[name, self._policies[name].as_dict()] for name in sorted(self._policies)]
        )

    def security_specs(self) -> dict[str, Any]:
        """Every tool's security spec as canonical JSON (null when absent), by name."""
        return {
            name: spec.as_dict() if (spec := self._security[name]) is not None else None
            for name in sorted(self._security)
        }

    def security_spec_sha256(self) -> str:
        """Hash of every tool's security spec (null when absent), by name; part of a secured episode's id."""
        return canonical_hash([[name, spec] for name, spec in self.security_specs().items()])

    def names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._policies

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    # A verify tool must be a registered read (registered earlier); alternatives must exist or be reserved.
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
