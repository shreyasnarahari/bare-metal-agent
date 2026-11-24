"""Bulkhead part 8: the AgentDojo adapter, pinned to one upstream revision.

AgentDojo is an evaluation dependency, never a runtime one: it is not in
``pyproject.toml``. The pinned release is ``agentdojo 0.1.35`` (git commit
``a75aba76``); the notebook installs exactly ``AGENTDOJO_REQUIREMENT``, and
``agentdojo_api()`` refuses any other installed version. Interfaces used, as
read from that revision's source:

- ``BasePipelineElement.query(query, runtime, env, messages, extra_args)``
  returning ``(query, runtime, env, messages, extra_args)``: the custom-pipeline
  extension point. ``TaskSuite.run_task_with_pipeline(pipeline, user_task,
  injection_task, injections)`` loads the (injected) environment, calls
  ``query`` once with the user task's prompt, and scores the returned
  environment and messages with the task's own ``utility`` and the injection
  task's ``security`` checks (``security`` true means the attacker's goal
  happened).
- ``FunctionsRuntime.run_function(env, name, kwargs) -> (result, error)``: the
  simulated tools. They mutate only the in-memory pydantic environment.
- ``Function.parameters.model_json_schema()``: a tool's arguments, exactly as
  AgentDojo's own OpenAI pipeline sends them (``_function_to_openai``).
- ``tool_result_to_str``: AgentDojo's YAML formatter for tool output.

The boundary. ``BulkheadPipeline`` runs one Steadyhand episode per ``query``:
the existing ``SteadyhandClient`` and Toolsmith service, native tool calls,
validation, guard, and recovery, and, in the secured configurations,
Bulkhead's policy gate and taint. Each AgentDojo tool becomes a Steadyhand tool
whose handler calls ``runtime.run_function`` on the episode's copy of the
benchmark environment (``EnvironmentStore``), so simulated side effects stay in
the benchmark's task state. Nothing runs on the host.

Schemas. A tool's AgentDojo schema is normalized by two declared rules only:
local ``$ref``s into ``$defs`` are inlined, and an object schema that does not
say ``additionalProperties`` is closed (pydantic ignores unknown keys, so
closing it refuses nothing a tool could use). An explicitly open object stays
open. Whatever Toolsmith's grammar compiler still cannot constrain makes the
tool ineligible, with its JSON Pointer (``toolsmith.evaluation.bfcl_adapter
.constraint_support``). An ineligible tool is withheld from every
configuration, and a case is ineligible when its user task's or injection
task's ground truth calls a withheld tool. Both lists are recorded with ids
and reasons; nothing is dropped silently.

Messages back to AgentDojo are rebuilt from the journal. A tool call appears
in an assistant message (and so in AgentDojo's function trace) only if its
handler ran: a call that validation, the guard, or the policy refused had no
effect, and the one trace-scored check in the pinned suites (Slack's
invite/add/remove sequence) asks whether the effect happened. The same rule
applies to all three configurations. The final assistant message always has
text content (an escalated episode's controller summary), so AgentDojo never
re-runs the pipeline.

Output trust per tool is declared in ``evaluation/data/bulkhead_agentdojo_v1.json``
and ``trust_check`` verifies it with canaries: every ground-truth call of every
task runs against an environment whose injection vectors hold canaries, and a
declared-trusted tool whose output carries one is a configuration error.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..types import canonical_hash
from .attacks import ROOT

AGENTDOJO_VERSION = "0.1.35"
AGENTDOJO_COMMIT = "a75aba7631d3ca5fb7ab938965c97ead2f9ff84b"
AGENTDOJO_REPOSITORY = "https://github.com/ethz-spylab/agentdojo"
AGENTDOJO_REQUIREMENT = f"agentdojo @ git+{AGENTDOJO_REPOSITORY}@{AGENTDOJO_COMMIT}"
DATA_PATH = ROOT / "evaluation" / "data" / "bulkhead_agentdojo_v1.json"
CONFIGURATIONS = ("baseline", "policy", "policy_taint")
APPROVAL_MODES = ("deny_all", "scripted")
EFFECTS = ("read", "idempotent_write", "non_idempotent_write")
PIPELINE_NAME = "bulkhead-local"  # AgentDojo's attacks map "local" to the victim name "Local model"
NOW = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)  # fixed approval clock: scripted approvals replay exactly
TOOL_TIMEOUT_S = 10.0


class AgentDojoPinError(RuntimeError):
    """AgentDojo is missing or is not the pinned release."""


class ClassificationError(ValueError):
    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("invalid AgentDojo classification:\n- " + "\n- ".join(self.problems))


# The pin


def installed_pin() -> dict[str, Any] | None:
    """The installed AgentDojo version and, for a git install, its commit; None when absent."""
    try:
        distribution = importlib.metadata.distribution("agentdojo")
    except importlib.metadata.PackageNotFoundError:
        return None
    commit = None
    direct = distribution.read_text("direct_url.json")
    if direct:
        try:
            commit = (json.loads(direct).get("vcs_info") or {}).get("commit_id")
        except ValueError:
            commit = None
    return {"version": distribution.version, "commit": commit}


def require_pinned() -> dict[str, Any]:
    pin = installed_pin()
    if pin is None:
        raise AgentDojoPinError(f"AgentDojo is not installed; install exactly: pip install '{AGENTDOJO_REQUIREMENT}'")
    if pin["version"] != AGENTDOJO_VERSION or (pin["commit"] is not None and pin["commit"] != AGENTDOJO_COMMIT):
        raise AgentDojoPinError(
            f"AgentDojo {pin['version']} (commit {pin['commit']}) is installed; this adapter is pinned to "
            f"{AGENTDOJO_VERSION} ({AGENTDOJO_COMMIT}). Install exactly: pip install '{AGENTDOJO_REQUIREMENT}'"
        )
    return pin


@dataclass(frozen=True)
class AgentDojoAPI:
    """The parts of AgentDojo the adapter and runner use; tests pass a stand-in with the same shapes."""

    FunctionCall: Any
    FunctionsRuntime: Any
    tool_result_to_str: Callable[[Any], str]
    system_message: str
    get_suites: Callable[[str], Mapping[str, Any]]
    load_attack: Callable[[str, Any, Any], Any]
    source: dict[str, Any]


def agentdojo_api() -> AgentDojoAPI:
    """The real, pinned AgentDojo. ``BulkheadPipeline`` is registered as a ``BasePipelineElement``."""
    pin = require_pinned()
    from agentdojo.agent_pipeline.agent_pipeline import load_system_message
    from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
    from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suites

    BasePipelineElement.register(BulkheadPipeline)
    return AgentDojoAPI(
        FunctionCall=FunctionCall, FunctionsRuntime=FunctionsRuntime, tool_result_to_str=tool_result_to_str,
        system_message=load_system_message(None), get_suites=get_suites, load_attack=load_attack,
        source={"package": "agentdojo", **pin},
    )


# Schemas and eligibility


def openai_tool(function: Any) -> dict[str, Any]:
    """AgentDojo's own conversion (``_function_to_openai``): name, description, pydantic JSON Schema."""
    return {
        "type": "function",
        "function": {"name": function.name, "description": function.description, "parameters": function.parameters.model_json_schema()},
    }


def normalize_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Inline local ``$ref``s and close objects that do not say ``additionalProperties``. Nothing else changes."""
    schema = copy.deepcopy(dict(schema))
    definitions = schema.pop("$defs", {})

    def resolve(node: Any, seen: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            reference = node["$ref"]
            name = reference.rsplit("/", 1)[-1]
            if not reference.startswith("#/$defs/") or name not in definitions or name in seen:
                return {key: resolve(value, seen) for key, value in node.items()}  # left for the compiler to refuse
            target = resolve(copy.deepcopy(definitions[name]), (*seen, name))
            for key, value in node.items():
                if key != "$ref":
                    target[key] = resolve(value, seen)
            node = target
        else:
            node = {key: resolve(value, seen) for key, value in node.items()}
        if node.get("type") == "object" and "additionalProperties" not in node:
            node["additionalProperties"] = False
        return node

    return resolve(schema, ())


@dataclass(frozen=True)
class ToolEligibility:
    tool: str
    eligible: bool
    definition_sha256: str
    pointer: str | None = None
    keyword: str | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "eligible": self.eligible, "definition_sha256": self.definition_sha256, "pointer": self.pointer, "keyword": self.keyword, "message": self.message}


def model_facing_tool(function: Any) -> dict[str, Any]:
    tool = openai_tool(function)
    tool["function"]["parameters"] = normalize_schema(tool["function"]["parameters"])
    return tool


def tool_eligibility(functions: Iterable[Any]) -> list[ToolEligibility]:
    """Each tool on its own through Toolsmith's grammar compiler; a refusal carries its JSON Pointer."""
    from toolsmith.evaluation.bfcl_adapter import constraint_support

    rows = []
    for function in functions:
        definition = model_facing_tool(function)
        support = constraint_support([definition])
        error = support.error or {}
        rows.append(ToolEligibility(function.name, support.eligible, canonical_hash(definition), error.get("pointer"), error.get("keyword"), error.get("message")))
    return rows


# The classification data


def validate_classification(data: Any) -> list[str]:
    problems: list[str] = []
    if not isinstance(data, Mapping) or set(data) != {"dataset", "version", "description", "agentdojo", "suites"}:
        return ["the classification has exactly the keys agentdojo, dataset, description, suites, version"]
    pin = data["agentdojo"]
    if (pin.get("version"), pin.get("commit")) != (AGENTDOJO_VERSION, AGENTDOJO_COMMIT):
        problems.append(f"agentdojo pin {pin.get('version')}/{pin.get('commit')} is not the adapter's {AGENTDOJO_VERSION}/{AGENTDOJO_COMMIT}")
    for name, suite in data["suites"].items():
        if set(suite) != {"tools", "policy"}:
            problems.append(f"{name}: a suite has exactly tools and policy")
            continue
        for tool, entry in suite["tools"].items():
            if set(entry) != {"effect", "output_trust", "sinks"}:
                problems.append(f"{name}.{tool}: a tool has exactly effect, output_trust, sinks")
                continue
            if entry["effect"] not in EFFECTS:
                problems.append(f"{name}.{tool}: effect must be one of {list(EFFECTS)}")
            if entry["output_trust"] not in ("trusted", "untrusted"):
                problems.append(f"{name}.{tool}: output_trust must be trusted or untrusted")
            if entry["sinks"] not in ([], ["external_effect"]):
                problems.append(f"{name}.{tool}: a simulated tool's only sink is external_effect")
            if entry["effect"] == "read" and entry["sinks"]:
                problems.append(f"{name}.{tool}: a read has no sink")
            if entry["effect"] != "read" and not entry["sinks"]:
                problems.append(f"{name}.{tool}: a write has the external_effect sink")
        policy = suite["policy"]
        if set(policy) != {"ask_tools", "deny_tools", "rationale"}:
            problems.append(f"{name}: policy has exactly ask_tools, deny_tools, rationale")
            continue
        unknown = (set(policy["ask_tools"]) | set(policy["deny_tools"])) - set(suite["tools"])
        if unknown:
            problems.append(f"{name}: policy names tools the suite does not classify: {sorted(unknown)}")
        if set(policy["ask_tools"]) & set(policy["deny_tools"]):
            problems.append(f"{name}: a tool cannot both ask and be denied")
    return problems


def load_classification(path: str | Path = DATA_PATH) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate_classification(data)
    if problems:
        raise ClassificationError(problems)
    return data


# The benchmark environment as an episode store


class EnvironmentStore:
    """A Steadyhand store backed by one AgentDojo task environment (a pydantic model) and its runtime.

    Handlers run under ``lock`` against ``env``; the hash covers the environment's full JSON dump.
    """

    def __init__(self, env: Any, runtime: Any, *, fixture_sha256: str | None = None) -> None:
        self.env = env
        self.runtime = runtime
        self.lock = threading.RLock()
        self.fixture_sha256 = fixture_sha256 or self.snapshot_sha256()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("lock")
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.lock = threading.RLock()

    def to_dict(self) -> dict[str, Any]:
        return self.env.model_dump(mode="json")

    def snapshot_sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def clone(self) -> EnvironmentStore:
        with self.lock:
            return EnvironmentStore(self.env.model_copy(deep=True), self.runtime, fixture_sha256=self.fixture_sha256)


class ToolError(RuntimeError):
    """The simulated tool reported an error (``run_function``'s ``"Type: message"`` text)."""


def make_handler(name: str, description: str, formatter: Callable[[Any], str]) -> Callable[..., str]:
    def handler(store: EnvironmentStore, **arguments: Any) -> str:
        result, error = store.runtime.run_function(store.env, name, arguments)
        if error is not None:
            raise ToolError(error)
        return formatter(result)

    handler.__name__ = handler.__qualname__ = name
    handler.__doc__ = description
    return handler


# One suite, adapted


class SuiteAdapter:
    """One AgentDojo suite as Steadyhand tools: eligibility, the frozen registry, and the three configurations."""

    def __init__(self, suite: Any, classification: Mapping[str, Any], api: AgentDojoAPI) -> None:
        from bulkhead.manifest import SecuritySpec, Sink, Trust
        from steadyhand.policy import Effect, ToolPolicy
        from steadyhand.registry import SteadyhandRegistry

        self.suite = suite
        self.api = api
        self.name = suite.name
        self._ground_truth: dict[tuple[str, str], set[str]] = {}
        if self.name not in classification["suites"]:
            raise ClassificationError([f"suite {self.name!r} is not classified"])
        declared = classification["suites"][self.name]
        self.classification = declared
        functions = list(suite.tools)
        names = [f.name for f in functions]
        missing = sorted(set(names) - set(declared["tools"]))
        extra = sorted(set(declared["tools"]) - set(names))
        if missing or extra:
            raise ClassificationError([f"{self.name}: tools not classified {missing}; classified but absent {extra}"])
        self.eligibility = tool_eligibility(functions)
        eligible = {row.tool for row in self.eligibility if row.eligible}
        self.withheld = [row for row in self.eligibility if not row.eligible]
        registry = SteadyhandRegistry()
        for function in functions:
            if function.name not in eligible:
                continue
            entry = declared["tools"][function.name]
            effect = Effect(entry["effect"])
            definition = model_facing_tool(function)["function"]
            registry.register(
                make_handler(function.name, definition["description"], api.tool_result_to_str),
                ToolPolicy(effect=effect, timeout_s=TOOL_TIMEOUT_S, idempotent=effect.idempotent),
                security=SecuritySpec(sandbox=None, output_trust=Trust(entry["output_trust"]), sinks=tuple(Sink(s) for s in entry["sinks"])),
                parameters=definition["parameters"], description=definition["description"],
            )
        self.registry = registry.freeze()
        self.tools = tuple(self.registry.names())
        self.sinks = tuple(t for t in self.tools if declared["tools"][t]["sinks"])

    def task_policy(self, configuration: str) -> Any:
        from bulkhead.policy import TaskPolicy
        from bulkhead.taint import Sink

        if configuration not in ("policy", "policy_taint"):
            raise ValueError(f"{configuration!r} has no task policy")
        policy = self.classification["policy"]
        withheld = {row.tool for row in self.withheld}
        taint = (Sink.EXTERNAL_EFFECT, Sink.FILESYSTEM_SENSITIVE, Sink.NETWORK) if configuration == "policy_taint" else ()
        return TaskPolicy(
            tools=tuple(t for t in self.tools if t not in policy["deny_tools"]),
            deny_tools=tuple(sorted(set(policy["deny_tools"]) | withheld)),
            ask_tools=tuple(t for t in policy["ask_tools"] if t in self.tools),
            taint_sensitive=taint,
        )

    def security_config(self, configuration: str, approval_mode: str = "deny_all", grants: Sequence[Any] = ()) -> Any:
        """None for the baseline; otherwise the declared policy (with taint for ``policy_taint``)."""
        from bulkhead.approvals import ApprovalMode
        from steadyhand.security import SecurityConfig

        if configuration not in CONFIGURATIONS:
            raise ValueError(f"unknown configuration {configuration!r}")
        if approval_mode not in APPROVAL_MODES:
            raise ValueError(f"unknown approval mode {approval_mode!r}")
        if configuration == "baseline":
            return None
        mode = ApprovalMode.SCRIPTED if approval_mode == "scripted" else ApprovalMode.DENY_ALL
        return SecurityConfig(self.task_policy(configuration), approval_mode=mode, grants=tuple(grants))

    def check_runtime(self, runtime: Any) -> None:
        """The runtime AgentDojo hands the pipeline must hold the very tools the registry adapted."""
        functions = runtime.functions
        for name in self.tools:
            if name not in functions:
                raise ClassificationError([f"{self.name}: the runtime lacks {name}"])
        current = {name: canonical_hash(model_facing_tool(functions[name])) for name in self.tools}
        expected = {row.tool: row.definition_sha256 for row in self.eligibility if row.eligible}
        if current != expected:
            raise ClassificationError([f"{self.name}: the runtime's tool schemas differ from the adapted ones"])

    def ground_truth_tools(self, task: Any) -> set[str]:
        """The tools a task's ground truth calls (cached per task id)."""
        key = (type(task).__name__, task.ID)
        if key not in self._ground_truth:
            env = self.suite.load_and_inject_default_environment({})
            if hasattr(task, "PROMPT"):
                env = task.init_environment(env)
            self._ground_truth[key] = {call.function for call in task.ground_truth(env)}
        return set(self._ground_truth[key])

    def case_eligibility(self, user_task: Any, injection_task: Any | None) -> tuple[bool, str | None]:
        withheld = {row.tool: row for row in self.withheld}
        for label, task in (("user task", user_task), ("injection task", injection_task)):
            if task is None:
                continue
            used = sorted(self.ground_truth_tools(task) & set(withheld))
            if used:
                row = withheld[used[0]]
                return False, f"{label} {task.ID} ground truth calls withheld tool {row.tool} (unsupported schema at {row.pointer}: {row.message})"
        return True, None

    def identity(self) -> dict[str, Any]:
        return {
            "suite": self.name,
            "schema_sha256": self.registry.schemas_sha256(),
            "policy_sha256": self.registry.policies_sha256(),
            "security_spec_sha256": self.registry.security_spec_sha256(),
            "tools": list(self.tools),
            "withheld": [row.as_dict() for row in self.withheld],
            "task_policy_sha256": {c: self.task_policy(c).sha256() for c in ("policy", "policy_taint")},
        }


# Journal -> AgentDojo messages


def _arguments(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def executed_calls(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every call whose handler ran, in journal order: model calls and policy-layer attempts."""
    calls = []
    for record in records:
        if record["event"] in ("tool_result", "policy_attempt") and record.get("handler_called"):
            calls.append({"step": record["step"], "tool_call_id": record["tool_call_id"], "tool": record["tool"], "arguments": _arguments(record["arguments"]), "via": "model" if record["event"] == "tool_result" else "policy"})
    return calls


def to_agentdojo_messages(records: Sequence[Mapping[str, Any]], api: AgentDojoAPI) -> list[dict[str, Any]]:
    """AgentDojo chat messages from a finished journal: only calls whose handler ran appear as tool calls."""
    started, finished = records[0], records[-1]
    text = lambda value: [{"type": "text", "content": value}]  # noqa: E731
    messages: list[dict[str, Any]] = []
    for message in started["messages"]:
        if message["role"] in ("system", "user"):
            messages.append({"role": message["role"], "content": text(message.get("content") or "")})
    results: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        if record["event"] == "tool_result" and (record["step"], record["tool_call_id"]) not in results:
            results[(record["step"], record["tool_call_id"])] = record
    attempts = [r for r in records if r["event"] == "policy_attempt" and r.get("handler_called")]
    for record in records:
        if record["event"] != "model_turn" or record["kind"] != "tool_calls":
            continue
        step = record["step"]
        calls, answers = [], []
        for call in (record.get("assistant_message") or {}).get("tool_calls") or []:
            result = results.get((step, call["id"]))
            if result is None or not result.get("handler_called"):
                continue
            function_call = api.FunctionCall(function=result["tool"], args=_arguments(result["arguments"]), id=call["id"])
            calls.append(function_call)
            answers.append({"role": "tool", "content": text(result["content"]), "tool_call_id": call["id"], "tool_call": function_call, "error": None if result["ok"] else result["content"]})
        for attempt in (a for a in attempts if a["step"] == step):
            function_call = api.FunctionCall(function=attempt["tool"], args=_arguments(attempt["arguments"]), id=f"{attempt['tool_call_id']}:policy:{attempt['seq']}")
            calls.append(function_call)
            answers.append({"role": "tool", "content": text(""), "tool_call_id": function_call.id, "tool_call": function_call, "error": None if attempt["ok"] else attempt.get("error_type")})
        if calls:
            messages.append({"role": "assistant", "content": None, "tool_calls": calls})
            messages.extend(answers)
    messages.append({"role": "assistant", "content": text(finished.get("final_content") or ""), "tool_calls": None})
    return messages


# The pipeline


@dataclass
class PipelineOutcome:
    episodes: list[Any] = field(default_factory=list)  # EpisodeResult, parent first
    journals: list[str] = field(default_factory=list)
    start_sha256: str = ""
    final_sha256: str = ""
    approvals_requested: int = 0
    approved: int = 0
    declined: int = 0
    replayed: list[bool] = field(default_factory=list)  # True: completed earlier, rebuilt by replay


class CaseClaimedElsewhere(RuntimeError):
    """Another worker holds this episode's claim."""


class BulkheadPipeline:
    """An AgentDojo pipeline element that runs one claimed Steadyhand episode per ``query``.

    One instance serves one (case, configuration). ``client_factory`` gives a fresh model
    client per episode (a scripted model restarts for an approved retry). ``approvals`` are
    the case's scripted one-shot approvals, used only with ``approval_mode="scripted"``.
    """

    name = PIPELINE_NAME

    def __init__(
        self,
        adapter: SuiteAdapter,
        configuration: str,
        client_factory: Callable[[], Any],
        directory: str | Path,
        *,
        approval_mode: str = "deny_all",
        approvals: Sequence[Mapping[str, Any]] = (),
        run_label: str | None = None,
        budgets: Any = None,
        server_parallel: int | None = None,
        context_size: int | None = None,
    ) -> None:
        if configuration not in CONFIGURATIONS:
            raise ValueError(f"unknown configuration {configuration!r}")
        if approval_mode not in APPROVAL_MODES:
            raise ValueError(f"unknown approval mode {approval_mode!r}")
        self.adapter = adapter
        self.configuration = configuration
        self.client_factory = client_factory
        self.directory = Path(directory)
        self.approval_mode = approval_mode
        self.approvals = list(approvals)
        self.run_label = run_label
        self.budgets = budgets
        self.server_parallel = server_parallel
        self.context_size = context_size
        self.calls = 0
        self.outcome: PipelineOutcome | None = None

    def _episode(self, store: EnvironmentStore, messages: list[dict[str, Any]], security: Any) -> tuple[Any, bool]:
        from steadyhand.journal import manifest_path_for
        from steadyhand.manifest import build_manifest, load_manifest
        from steadyhand.recovery import no_sleep
        from steadyhand.replay import replay_episode, run_claimed_episode

        client = self.client_factory()
        options = dict(budgets=self.budgets, run_label=self.run_label, server_parallel=self.server_parallel, context_size=self.context_size)
        result = run_claimed_episode(self.directory, client, self.adapter.registry, store, messages, sleep=no_sleep, security=security, **options)
        if result is not None:
            return result, False
        manifest = build_manifest(
            client.settings(), self.adapter.registry, store, messages, budgets=(self.budgets.as_dict() if self.budgets else _default_budgets()),
            guard_limit=_guard_limit(), run_label=self.run_label, server_parallel=self.server_parallel, context_size=self.context_size, security=security,
        )
        path = self.directory / f"{manifest.episode_id}.jsonl"
        if not (manifest_path_for(path).exists() and load_manifest(manifest_path_for(path)).completed):
            raise CaseClaimedElsewhere(str(path))
        return replay_episode(path, registry=self.adapter.registry, store=store), True

    def query(self, query: str, runtime: Any, env: Any = None, messages: Sequence[Any] = (), extra_args: Mapping[str, Any] | None = None) -> tuple[Any, ...]:
        from bulkhead.approvals import ScriptedApproval, ScriptedApprover
        from steadyhand.journal import read_journal
        from steadyhand.recovery import no_sleep
        from steadyhand.security import approval_requests, resolve_approvals, run_approved_episode

        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("a BulkheadPipeline serves exactly one query")
        self.adapter.check_runtime(runtime)
        start = env.model_copy(deep=True)
        opening = [{"role": "system", "content": self.adapter.api.system_message}, {"role": "user", "content": query}]
        security = self.adapter.security_config(self.configuration, self.approval_mode)
        store = EnvironmentStore(start.model_copy(deep=True), runtime)
        outcome = PipelineOutcome(start_sha256=store.snapshot_sha256())
        result, replayed = self._episode(store, opening, security)
        outcome.episodes.append(result)
        outcome.journals.append(str(result.journal_path))
        outcome.replayed.append(replayed)
        if security is not None and result.status.value == "escalated":
            outcome.approvals_requested = len(approval_requests(result.journal_path))
            if self.approval_mode == "scripted" and outcome.approvals_requested:
                approver = ScriptedApprover(tuple(ScriptedApproval(a["tool"], a["arguments"], a.get("rule_id"), a["approve"]) for a in self.approvals))
                grants = resolve_approvals(result.journal_path, approver, now=NOW)
                outcome.approved, outcome.declined = len(grants), outcome.approvals_requested - len(grants)
                if grants:
                    store = EnvironmentStore(start.model_copy(deep=True), runtime)
                    child = run_approved_episode(self.directory, result.journal_path, grants[0], self.client_factory(), self.adapter.registry, store, now=NOW, sleep=no_sleep)
                    replayed = child is None
                    if child is None:
                        child = self._replay_child(result.journal_path, grants[0], store)
                    outcome.episodes.append(child)
                    outcome.journals.append(str(child.journal_path))
                    outcome.replayed.append(replayed)
            else:
                outcome.declined = outcome.approvals_requested
        outcome.final_sha256 = store.snapshot_sha256()
        self.outcome = outcome
        records = read_journal(outcome.journals[-1])
        return query, runtime, store.env, to_agentdojo_messages(records, self.adapter.api), dict(extra_args or {})

    def _replay_child(self, parent: str | Path, grant: Any, store: EnvironmentStore) -> Any:
        """An approved retry that completed earlier: find its journal by parent and grant, and replay it."""
        from steadyhand.journal import read_journal
        from steadyhand.manifest import load_manifest
        from steadyhand.replay import replay_episode

        parent_id = read_journal(parent)[0]["episode_id"]
        for manifest_file in sorted(self.directory.glob("*.manifest.json")):
            manifest = load_manifest(manifest_file)
            if manifest.parent_episode_id == parent_id and manifest.run_label == f"approved-{grant.approval_id}" and manifest.completed:
                return replay_episode(self.directory / f"{manifest.episode_id}.jsonl", registry=self.adapter.registry, store=store)
        raise CaseClaimedElsewhere(f"approved retry of {parent_id}")


def _default_budgets() -> dict[str, Any]:
    from steadyhand.controller import Budgets

    return Budgets().as_dict()


def _guard_limit() -> int:
    from steadyhand.controller import DEFAULT_GUARD_LIMIT

    return DEFAULT_GUARD_LIMIT


def replay_effects(start: Any, runtime: Any, calls: Sequence[Mapping[str, Any]]) -> str:
    """The environment hash after re-running exactly ``calls`` on a copy of ``start``."""
    env = start.model_copy(deep=True)
    for call in calls:
        runtime.run_function(env, call["tool"], call["arguments"])
    return canonical_hash(env.model_dump(mode="json"))


# Trust declarations, checked with canaries


def trust_check(adapter: SuiteAdapter) -> dict[str, Any]:
    """Run every task's ground truth on a canary-injected environment; report each tool's canary exposure.

    A declared-trusted tool whose output carried a canary is a problem. Trusted tools that no ground
    truth exercised are listed as unverified.
    """
    suite, api = adapter.suite, adapter.api
    canaries = {vector: f"<BULKHEAD-CANARY-{index}>" for index, vector in enumerate(sorted(suite.get_injection_vector_defaults()))}
    base = suite.load_and_inject_default_environment(canaries)
    runtime = api.FunctionsRuntime(suite.tools)
    exposure: dict[str, set[str]] = {}
    exercised: set[str] = set()
    tasks = [*suite.user_tasks.values(), *suite.injection_tasks.values()]
    for task in tasks:
        env = base.model_copy(deep=True)
        if hasattr(task, "PROMPT"):
            env = task.init_environment(env)
        for call in task.ground_truth(env.model_copy(deep=True)):
            result, error = runtime.run_function(env, call.function, dict(call.args))
            output = (api.tool_result_to_str(result) if error is None else "") + (error or "")
            exercised.add(call.function)
            carried = {vector for vector, canary in canaries.items() if canary in output}
            if carried:
                exposure.setdefault(call.function, set()).update(carried)
    declared = adapter.classification["tools"]
    trusted = {name for name, entry in declared.items() if entry["output_trust"] == "trusted"}
    problems = sorted(f"{tool} is declared trusted but its output carried {sorted(vectors)}" for tool, vectors in exposure.items() if tool in trusted)
    return {
        "suite": adapter.name,
        "vectors": sorted(canaries),
        "exposed_tools": {tool: sorted(vectors) for tool, vectors in sorted(exposure.items())},
        "unverified_trusted": sorted(trusted - exercised),
        "problems": problems,
    }


__all__ = [
    "AGENTDOJO_COMMIT",
    "AGENTDOJO_REQUIREMENT",
    "AGENTDOJO_VERSION",
    "APPROVAL_MODES",
    "CONFIGURATIONS",
    "AgentDojoAPI",
    "AgentDojoPinError",
    "BulkheadPipeline",
    "CaseClaimedElsewhere",
    "ClassificationError",
    "EnvironmentStore",
    "PipelineOutcome",
    "SuiteAdapter",
    "ToolEligibility",
    "agentdojo_api",
    "executed_calls",
    "installed_pin",
    "load_classification",
    "normalize_schema",
    "openai_tool",
    "replay_effects",
    "require_pinned",
    "to_agentdojo_messages",
    "tool_eligibility",
    "trust_check",
    "validate_classification",
]
