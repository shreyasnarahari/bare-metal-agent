# Days 4–5: Compile JSON Schemas into GBNF grammars

## Context

The project now has a locally tested Qwen3 parser, streaming transport, OpenAI delta adapter, and FastAPI shim. `LlamaCppClient.complete()` and `stream_complete()` already forward a `grammar` string (`src/toolsmith/backends/llama_cpp.py:77`), but `ToolsmithService` never constructs or passes one (`src/toolsmith/api.py:82`). Tool registration produces Draft 2020-12 schemas, including optional enum properties such as `get_weather(city, unit="c")` (`src/toolsmith/registry.py:32`). The endpoint currently accepts only `tool_choice="auto"` and does not validate parsed arguments against the selected tool.

The outcome for these two evenings is a strict JSON-Schema-to-GBNF compiler, Qwen3 tool-call envelope generation, all four OpenAI `tool_choice` modes, semantic post-validation, and real verification with both llama.cpp's grammar validator and Qwen3-8B. Prompt-only output remains available as an ablation baseline. Grammar validity, schema validity, and semantic correctness will be measured and reported separately.

The Days 3–4 implementation is locally tested but still lacks its real Qwen/Colab checkpoint. A fresh full run of the canonical notebook for this milestone will rerun those existing streaming checks rather than skipping them.

## Approach

Build a small, deterministic compiler owned by Toolsmith; do not wrap llama.cpp's converter and call it ours. Use llama.cpp's pinned converter and GBNF validator as independent oracles. Compile only a documented subset and reject everything else with a schema path. Then thread the compiled grammar through both existing completion paths without changing the parser's public API.

### Evening 4 — compiler and offline correctness (CPU)

1. Write the schema support contract and shared valid/invalid corpus.
2. Implement JSON primitives, escaping, arrays, enums, and nested closed objects.
3. Add the Qwen3 envelope and `tool_choice` grammar variants.
4. Prove deterministic output and rejection behavior with unit tests.

**Evening 4 exit gate:** all local compiler and API-independent tests pass; every unsupported fixture fails loudly; representative generated grammars are ready for llama.cpp validation.

### Evening 5 — integration and real engine/model checks (CPU, then L4)

1. Integrate grammar selection into non-streaming and streaming service calls.
2. Add argument and `tool_choice` post-validation, without a repair retry.
3. Validate complete strings with llama.cpp's actual GBNF implementation and compare against its converter.
4. Extend `notebooks/01_toolsmith.ipynb` with offline, validator, SDK, and ablation cells.
5. Run the entire canonical notebook on a fresh L4 runtime and save evidence to Drive.

**Evening 5 exit gate:** the full local suite passes, llama.cpp accepts/rejects the corpus correctly, real constrained calls parse and validate, all `tool_choice` modes work through the official OpenAI SDK, and cleanup remains the final notebook cell.

## Key changes

### 1. Strict schema compiler

Add `src/toolsmith/grammar.py` with:

- `SchemaNotSupportedError`, including the unsupported keyword and JSON Pointer such as `/properties/age/minimum`.
- `GrammarConfigurationError` for invalid tool/choice combinations.
- An immutable `CompiledGrammar` containing the GBNF text, selected mode, tool names, and stable grammar hash for logs.
- `GBNFCompiler.compile_schema(schema)` for a standalone JSON value.
- `GBNFCompiler.compile_tools(tools, tool_choice)` for Qwen3 tool calling.
- An internal rule builder that assigns sequential rule names (`schema-0`, `tool-0-args`, etc.) rather than sanitizing arbitrary property or tool names into colliding GBNF identifiers.

Supported Draft 2020-12 subset:

- `object` with `properties`, `required`, and explicit `additionalProperties: false`.
- Required and optional properties, each emitted at most once. Use deterministic declaration order, with required properties before optional ones; nested optional-suffix rules must permit skipped optional properties without trailing commas or duplicates.
- `string`, `integer`, `number`, `boolean`, and `null`.
- Homogeneous `array` with a required inline `items` schema, including empty and nested arrays.
- `enum` containing finite JSON scalar values. Compile exact JSON literals, not unquoted approximations.
- Inline nesting of the supported object/array schemas.
- Annotation-only `title`, `description`, `default`, `examples`, and the Draft 2020-12 `$schema` declaration. Ignoring these must not weaken validation; `default` does not make a property optional by itself.

Reject, rather than silently ignore:

- Missing/`true`/schema-valued `additionalProperties`, boolean schemas, and schemas without a supported type or exact `enum`.
- `$ref`, `$defs`, remote references, `allOf`, `anyOf`, `oneOf`, `not`, conditionals, and type unions.
- `pattern`, `format`, length/range limits, `multipleOf`, array size/uniqueness constraints, object size constraints, `patternProperties`, `propertyNames`, tuple/`prefixItems`, and unsupported `items` forms.
- Malformed required lists, required names absent from `properties`, contradictory `type`/`enum` combinations, non-finite numbers, and recursion or inputs exceeding configured depth/rule/tool limits.

Use JSON Pointer-rich errors and map them to HTTP 400 before any model request. Validate input mappings with `jsonschema.Draft202012Validator.check_schema()` first, then apply the stricter compiler contract.

Implement JSON syntax carefully:

- A single GBNF literal-escaping helper for quotes, backslashes, controls, newlines, and Unicode.
- JSON strings with valid quoted characters and only valid JSON escapes (`\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`, and `\uXXXX`); reject unescaped control characters.
- Integers as `-? (0 | [1-9][0-9]*)`; reject leading zeros and booleans masquerading as integers.
- Numbers with JSON decimal/exponent syntax; reject `NaN`, infinities, `+1`, `01`, `1.`, and incomplete exponents.
- Bounded JSON whitespace rules. Token limits remain the outer bound for arbitrary strings/text, and truncation remains an explicit error.

Add focused tests in `tests/test_grammar.py` and reusable cases in `tests/fixtures/grammar_cases.json`. Cover empty/all-required/all-optional/mixed objects, skipped optionals, duplicate keys, nested values, arrays, all primitive types, scalar enums, escaped keys/tool names, Unicode, literal protocol-looking text inside JSON strings, deterministic regeneration, unique rule names, and every rejection class.

### 2. Qwen3 envelopes and `tool_choice`

Use the same protocol the existing parser understands:

```text
<tool_call>
{"name":"get_weather","arguments":{...}}
</tool_call>
```

Generate one argument rule per registered tool and reference that rule only from that tool's envelope. Fix envelope fields to `name` followed by `arguments`; reject extra envelope fields and calls using another tool's argument schema. Permit bounded whitespace and multiple adjacent calls where the selected mode allows them.

Implement these modes:

| `tool_choice` | Grammar and postcondition |
| --- | --- |
| omitted/`null`/`auto`, with tools | Either non-empty plain text or one or more calls to any registered tool |
| `none` | Non-empty plain text only; zero calls |
| `required` | One or more calls to any registered tool |
| named function | Exactly one call to that named tool |

For no registered tools, omitted/`auto`/`none` use the existing unconstrained text path; `required` and named choices are request errors. Validate named-function request shape and reject unknown names before generation.

Do **not** use an unrestricted `.*`-style text alternative in `auto`: it would also accept malformed tool-call markup and defeat the guarantee. The text production must conservatively exclude Qwen protocol openings/closings while allowing ordinary text and Unicode. Constrained tool responses are calls plus permitted inter-call whitespace, not arbitrary prose before/between/after calls. The existing parser still supports mixed prose for prompt-only baselines.

For this milestone, constrained requests with tools require `chat_template_kwargs.enable_thinking=false`. Reject enabled/unspecified thinking for those requests with a clear 400 error instead of silently dropping constraints or changing the prompt. No-tool text requests remain unchanged. Trigger-activated, think-then-constrain decoding belongs to the later custom-sampler milestone.

Keep `response_format` explicitly unsupported for now; `compile_schema()` is nevertheless tested standalone so the later endpoint can reuse it.

### 3. Service integration and semantic validation

Add `src/toolsmith/validation.py` with a small validator that:

- Looks up each parsed call by name and rejects unknown tools.
- Parses argument JSON and validates it with a cached `Draft202012Validator` for that exact tool.
- Enforces zero/at-least-one/exactly-one/named-tool postconditions independently of the grammar.
- Returns structured errors suitable for the existing `APIError` and SSE error paths.
- Never executes the tool and does not attempt repair.

Refactor `ToolsmithService._prepare()` to return one internal prepared-request object containing the registered tools, rendered prompt, normalized choice, and optional compiled grammar. This avoids divergent non-streaming/streaming decisions.

Update both paths in `src/toolsmith/api.py`:

- Pass `grammar=prepared.grammar.text` to `client.complete()` and `client.stream_complete()`; no-tools requests omit the field as before.
- Validate parsed responses before returning a successful non-streaming response or emitting a successful stream finish chunk.
- On model-output validation failure, return/emit an explicit server error, never `finish_reason="tool_calls"`; close upstream streaming iterators and retain cancellation behavior.
- Record grammar text/hash, normalized choice, validation outcome, raw output, and timing metadata through the existing redacting `JsonlRecorder`.

Extend `tests/test_api.py` and `tests/test_streaming_api.py` to assert identical grammar selection in both paths, no grammar for no-tools text, all choice modes, request errors before backend calls, unknown tools, schema-invalid arguments, postcondition violations, mid-stream validation errors without false success, and unchanged truncation/disconnect handling. Add focused validator tests in `tests/test_validation.py`.

### 4. Independent llama.cpp validation and differential comparison

Extend only `notebooks/01_toolsmith.ipynb`; do not create another notebook or alter the shared cleanup design.

Add these cells before `cleanup`:

1. **`grammar-notes`** — explain syntactic versus schema versus semantic validity, supported keywords, fixed property order, and why `auto` needs marker-safe text.
2. **`grammar-offline-corpus`** — compile every supported fixture, assert deterministic output and expected unsupported-schema errors, run `jsonschema` on valid decoded values, and save the generated grammars/corpus results.
3. **`grammar-validator-comparison`** — ensure the pinned `b5401` source is present without destructive reset; configure a separate CPU build with tests enabled and build `test-gbnf-validator`. Run each complete input against our grammar. The pinned validator prints its verdict but returns zero even for invalid input, so parse the explicit verdict text and treat grammar-initialization failures separately. Invoke `examples/json_schema_to_grammar.py` on the same standalone schemas, validate the same corpus with its grammar, capture warnings, and record differences. Require language-equivalent expected accept/reject behavior on the shared corpus, not byte-identical GBNF.
4. **`grammar-tool-choice-sdk`** — through the official OpenAI SDK, test `auto` text, `auto` tool call, `none`, `required`, and named function in non-streaming mode; repeat required/named and one malformed-stream error case in streaming mode. Reconstruct chunks semantically and assert call counts, selected names, arguments, and finish/error behavior.
5. **`grammar-ablation`** — render one fixed prompt once and compare (a) prompt-only plus parser and (b) the same prompt with our GBNF at fixed seed/settings. Include simple, nested/array/enum, parallel, and no-tool-needed prompts. Record raw responses, parse rate, schema-valid rate, choice correctness, false-call count, truncation, elapsed time, token counts/tokens per second, grammar size/rule count, and compile time. Do not manufacture baseline failures or claim semantic/BFCL accuracy from this small matrix.

Save generated reference and Toolsmith grammars under `MyDrive/toolsmith/grammars/days-04-05/` and JSONL evidence as:

```text
MyDrive/toolsmith/records/days-04-05-offline-corpus.jsonl
MyDrive/toolsmith/records/days-04-05-validator-comparison.jsonl
MyDrive/toolsmith/records/days-04-05-tool-choice.jsonl
MyDrive/toolsmith/records/days-04-05-ablation.jsonl
```

Update `tests/test_notebook_01.py` to validate the new cell IDs, execute the offline grammar cell in a temporary directory, check saved records/redaction, and retain the existing canonical-notebook/no-stale-error/no-destructive-sync checks. Keep `cleanup` last and preserve all serving/cache fixes and previous streaming cells.

## Verification

Local, using the project's existing environment:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q src tests
git diff --check
```

Also validate notebook JSON, unique cell IDs, Python syntax, no saved error outputs, and that `cleanup` remains last. Check `git status --short --branch` before editing and after verification; do not discard user edits or commit/push unless separately requested.

Real Colab acceptance on a fresh L4 runtime:

1. Make the current working tree available in `/content/bare-metal-agent` and run `notebooks/01_toolsmith.ipynb` from top to bottom.
2. All previous health, raw streaming equality, parser replay, and OpenAI streaming checks must still pass.
3. Every valid supported corpus string is accepted by our GBNF; every invalid string is rejected. Valid decoded values also pass `jsonschema`.
4. Every unsupported schema fails before generation with its keyword and JSON Pointer. Our compiler must not silently under-constrain any shared differential case; record upstream differences rather than hiding them.
5. For every accepted constrained Qwen generation in the matrix, the existing parser succeeds, selected calls satisfy the requested choice, and arguments validate against the named tool. Named mode returns exactly one requested call; `none` returns none; `required` returns at least one.
6. Streaming and non-streaming requests send equivalent grammars and preserve usage, terminal error, truncation, and cleanup behavior.
7. All four new evidence files and the generated `.gbnf` artifacts exist on Drive and contain no HF token or local API key.
8. Report small-sample measurements as observations, not guarantees. A malformed prompt-only baseline is useful evidence, but its absence does not invalidate the grammar tests.

If Colab fails, preserve the first failing cell output, raw model text, generated grammar, and relevant server-log tail as fixtures; make a targeted compiler/integration fix and rerun the full suite and notebook. Do not weaken negative grammar cases or truncation checks to accommodate model behavior.

## Assumptions and boundaries

- Draft 2020-12 is the input dialect; the strict subset above is intentional for this milestone. Inline nesting is supported, but `$ref`/recursive Pydantic models and unions are deferred and must fail loudly.
- Fixed property order is a generation restriction, not permission to emit schema-invalid JSON. This tradeoff keeps optional-property grammars small and deterministic.
- Thinking-enabled constrained generation is deferred to the trigger-activated sampler work; constrained tool requests explicitly disable thinking for now.
- No token-masking sampler, vocabulary trie, C++ port, Qwen3.5 XML format, repair retry, tool execution, agent state machine, BFCL run, JSONSchemaBench run, or new dependency is part of these two evenings.
- The current uncommitted Days 3–4 and canonical-notebook changes are preserved. Real Qwen acceptance remains outstanding until the full notebook run passes.

Pinned implementation references:

- [llama.cpp b5401 GBNF guide](https://github.com/ggml-org/llama.cpp/blob/b5401/grammars/README.md)
- [llama.cpp b5401 JSON Schema converter](https://github.com/ggml-org/llama.cpp/blob/b5401/examples/json_schema_to_grammar.py)
- [llama.cpp b5401 GBNF validator](https://github.com/ggml-org/llama.cpp/blob/b5401/tests/test-gbnf-validator.cpp)
