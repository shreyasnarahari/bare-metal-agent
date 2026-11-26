# Part 5 — Long-term memory storage and retrieval

**Track:** Tidemark, Day 4 of 10
**Milestone:** M4, evening 8
**Depends on:** `04-cited-compaction-and-replay.md`
**Produces:** a namespace-scoped SQLite/FTS5 memory store, a small local embedding provider, hybrid retrieval, supersession, deletion, and durable backups

## Outcome

Persist round-sized facts across sessions with provenance, timestamps, trust, and explicit replacement semantics. Retrieval combines lexical and vector evidence with deterministic ranking and time filters. A newer fact supersedes an older one instead of allowing both to outvote each other, and forgetting removes content from every index without losing the tombstone needed for safe restore.

This part is a library with no agent-loop or model-tool side effects. Part 6 puts it behind Bulkhead and replay. No memory is retrieved into the prompt automatically here.

## What already exists

- Part 4 defines canonical `<episode_id>:<seq>` source refs, trust-bearing source evidence, and citation validation.
- Bulkhead defines `Trust`, taint-sensitive sinks, approvals, and policy/spec hashing. The storage library records labels; it does not decide ALLOW/DENY/ASK itself.
- Part 2's pinned block can change only at session/epoch boundaries. Therefore retrieval runs at explicit boundaries, not as an implicit different SQLite query on every render.
- `pyproject.toml` has no SQLite or embedding package beyond Python's standard `sqlite3`; `torch` and `transformers` are already in the `colab` extra. Do not add a hosted memory/vector service or assume `sentence-transformers`, FAISS, or hnswlib is installed.
- The repository's durability pattern is fast local files plus atomic Drive mirrors. A live SQLite database and its WAL must not be treated as independently copyable Drive files.

## What this part adds

```text
src/tidemark/memory_types.py            # MemoryRecord, SearchQuery, SearchHit, Trust imports
src/tidemark/memory_store.py            # SQLite schema, transactions, backup/restore, maintenance
src/tidemark/memory_retrieval.py        # Time parsing, FTS/vector candidate fusion, ranking
src/tidemark/embeddings.py              # EmbeddingProvider protocol and local HF implementation
src/tidemark/memory_backup.py           # Consistent snapshots, manifests, integrity checks
src/tidemark/eval/memory_metrics.py     # Recall@k, NDCG@k, supersession/deletion checks
tests/test_tidemark_memory_store.py
tests/test_tidemark_memory_retrieval.py
tests/test_tidemark_embeddings.py
tests/test_tidemark_memory_backup.py
tests/fixtures/tidemark_memory/
```

Core storage uses the standard library and existing `numpy`-free vector math if necessary; small exact vector indexes do not justify a new dependency. The real embedding provider uses the already declared Colab `torch`/`transformers` extras and a pinned model revision. CPU unit tests use a deterministic fake embedder and never present its scores as model retrieval quality.

## Memory record contract

Store one round-sized value per record, not a whole undifferentiated session transcript:

```text
memory_id, namespace, family_id, revision,
key, key_variants[], value,
created_at, valid_from, valid_to, source_refs[],
trust, supersedes, status, schema_version,
embedding_model, embedding_revision, content_sha256
```

- `namespace` is assigned by trusted server configuration and scopes every key, query, backup, and deletion.
- `key` is the canonical short lookup key; `key_variants` contains deterministic aliases/entities and optional explicitly recorded extracted facts. `value` remains the round-level evidence/fact.
- `family_id` identifies one supersession chain and `revision` increases within it. `supersedes` points at the currently active predecessor or is null.
- `created_at` records when the fact was stored; `valid_from`/`valid_to` describe when it applies. Do not conflate them. All timestamps are ISO-8601 UTC or dataset-provided logical times.
- `source_refs[]` points to authorized journal events. Trust is `trusted` or `untrusted`, inherited from the episode/content, never supplied as a model assertion.
- `status` is `active`, `superseded`, or `forgotten`. Forgotten rows retain only minimal tombstone metadata, not the deleted value or key variants.
- `content_sha256` covers canonical key/value/source/validity fields and detects backup/index drift.

The model-facing write schema in part 6 is much smaller: key, value, optional aliases, validity, source refs, and optional supersedes. Namespace, trust, memory id, revision, embedding identity, and storage timestamps are server-derived.

### Deterministic identity and writes

Derive `memory_id` from namespace, canonical key/value, source refs, validity, and family linkage. An identical authorized retry returns the existing receipt without adding a second fact or supersession edge. A changed value or source gets a new id/revision.

Supersession is one transaction that:

1. Checks namespace and that the predecessor is active and belongs to the same authorized family/domain.
2. Inserts the new record and source links.
3. Marks the old record superseded by the new id/revision.
4. Updates FTS and the vector row for the new record; removes the old record from active candidate queries while retaining it for audit.
5. Commits all rows before returning a receipt.

An expected-precondition/revision check prevents two workers from superseding the same old fact concurrently. There is no silent last-writer-wins update.

## SQLite schema and transactions

Use a versioned migration with at least:

```text
memory_meta(schema_version, namespace, config_sha256, next_revision, updated_at)
memories(... current record fields, status, superseded_by ...)
memory_sources(memory_id, episode_id, seq, relation)
memory_fts USING fts5(key_variants, value, content='memories', content_rowid=rowid)
memory_vectors(memory_id, dimension, dtype, model_revision, vector_blob)
memory_tombstones(memory_id, namespace, content_sha256, forgotten_at, reason_code, source_refs)
memory_operations(sequence, operation, memory_id, revision, request_sha256, result_sha256)
```

Enable `PRAGMA foreign_keys=ON`, a bounded busy timeout, and local WAL/synchronous settings appropriate for crash recovery. Serialize writers per database or use one writer connection with a lock; readers never share a connection concurrently. All SQL values are parameters. FTS queries are constructed from escaped terms, not raw user-supplied `MATCH` syntax.

`memory_operations` is the database's monotonic mutation log and supports resume revision checks. It stores hashes and provenance, not another unbounded copy of every value.

### FTS5 capability probe

At startup, execute a real temporary `CREATE VIRTUAL TABLE ... USING fts5`, insert/query/delete a canary, and record SQLite/library versions and tokenizer support. Failure produces `memory_unavailable:fts5` for the real-memory configuration. Do not silently substitute a slow table scan and then call it the same retrieval algorithm.

A non-FTS `LIKE` implementation may exist only as an explicitly named unit-test reference for ranking differential tests, never as the live fallback.

## Embedding provider

Define:

```python
class EmbeddingProvider(Protocol):
    @property
    def identity(self) -> EmbeddingIdentity: ...
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]: ...
    def embed_query(self, text: str) -> Vector: ...
```

Start with `BAAI/bge-small-en-v1.5` or a measured Qwen3-0.6B embedding alternative. Pin repository and full revision, pooling, query instruction, normalization, dtype, dimension, maximum sequence length, and device. Validate the actual model card/config before choosing pooling; do not infer it from a model-name substring.

The default local-HF path uses `AutoTokenizer`/`AutoModel` from the existing Colab stack, batches deterministically, applies the documented query/document prompts, mean/CLS pooling as appropriate, and L2-normalizes vectors. It runs on CPU by default so the agent GGUF remains the only large resident GPU model. Sharing the GPU is a separate measured configuration after a VRAM probe.

Vectors are little-endian float32 blobs with dimension/model checks. For the small store, compute exact cosine similarity in deterministic memory-id order. An approximate index is a later optimization and must be differential-tested against exact results.

Tests use `FakeEmbeddingProvider` with declared vectors. `HashEmbedder`, if used for smoke fixtures, is labeled non-semantic and excluded from recall claims.

## Hybrid retrieval

`MemorySearch` contains:

```text
namespace, query, k, as_of, time_range, trust_filter,
key_filter, include_superseded=false, ranking_config
```

`k` is server-capped (default maximum 20; the tools in part 6 normally request 5 or 10). `as_of` comes from the task/dataset or trusted session configuration, never from an invented model time range. Obvious date phrases are parsed by deterministic rules first; a model query rewriter is a separate versioned ablation, not a prerequisite.

Candidate generation:

1. Filter namespace, tombstones, status, key/trust/time constraints in SQL before scoring. Active records only by default; audit callers must explicitly request superseded history.
2. Retrieve a bounded FTS set ranked by BM25 using the probed tokenizer and escaped query terms.
3. Embed the query once and retrieve a bounded exact-cosine set over eligible active vectors with the same model revision.
4. Fuse using a fixed reciprocal-rank rule, for example `score = Σ 1/(60 + rank)` over both lists. Tie-break by descending score, descending recency where declared, then ascending memory id.
5. Return each hit with value, key variants, timestamps, source refs, trust, revision, and lexical/vector/fused scores.

Keep ranking weights/window/candidate limits in `RankingConfig` and hash them into the run. Do not tune weights on the evaluation questions and then report recall on the same ids. FTS-only, vector-only, and fused are separate comparison modes.

If one leg is unavailable, return a typed degraded result only when the configuration explicitly allows that leg to be absent. The result and manifest name the degraded mode; it is not silently compared with full hybrid retrieval.

### Retrieval into context

Part 6 uses a search result to build one canonical pinned-memory block at session start or a declared epoch. Mid-session explicit `memory_search` results remain ordinary recent tool observations. The assembler never calls SQLite during `render`.

An untrusted stored fact stays labeled untrusted in the result and prompt block. A database write or human approval does not launder it into trusted content. Part 6 applies the corresponding episode taint before any subsequent sensitive action.

## Supersession, deletion, and retention

### Query behavior

- Active search excludes superseded and forgotten records before either ranking leg.
- `get_memory(id, include_history=True)` can show an audit chain, but the model-facing ordinary search does not resurrect old values.
- A newer value does not delete history unless forget/retention says so; supersession is a knowledge update, not erasure.

### Forget behavior

`forget(memory_id, reason_code, source_refs)` is a transaction that:

- verifies namespace and current record identity;
- removes value, key variants, FTS rows, vectors, and source-text copies covered by the deletion contract;
- writes a tombstone containing id, namespace, old content hash, deletion time/reason, and authorized refs, but no deleted value;
- prevents restore/retry of an old operation from silently resurrecting that same id;
- records the mutation revision and receipt.

Deleting one fact does not accidentally delete unrelated facts sharing a source event. A namespace purge is a separate privileged maintenance operation with its own explicit approval and is outside the model tool.

Retention scans use valid/retention timestamps and the declared policy. They never run implicitly during render or search, and they must account for open episode/checkpoint references before deleting required blobs or memory history.

## CPU/GPU indexing and retrieval pipeline

Memory work must overlap agent generation without changing ranking or operation order:

- Run one shared embedding service per configured CPU worker, with the model loaded once and bounded input/output queues. Start with one process on T4 and tune Torch threads against llama.cpp's two-vCPU budget; on L4, test a larger thread allocation before considering another model process and its RAM cost.
- Batch independent documents from upcoming indexing jobs and authorized writes. Pin batch size, padding, dtype, pooling, and model revision in the manifest because batching can change numerical results. Differential-test every candidate batch size against serial vectors before using its speedup; replay uses recorded vectors/results.
- Precompute LongMemEval/document embeddings and exact vector indexes on CPU while the GPU runs unrelated model phases. Cache them by namespace + canonical content + preprocessing + embedding identity + batch configuration. Never reuse embeddings across different models or preprocessing rules.
- Overlap FTS inserts and candidate generation with other episodes' GPU calls. A dedicated writer queue preserves memory mutation order and short transactions; readers use their own connections/views. Compute embeddings outside the writer transaction.
- Bound candidate lists, vector scratch memory, and prefetched result blocks by `ResourceBudget`. On T4, avoid duplicate embedding models and large process pools that exhaust roughly 13 GB RAM; on L4, scale only while tasks/minute improves and RAM headroom remains.
- The default keeps embeddings on CPU and the agent GGUF alone on GPU. An optional shared-GPU embedding probe may run only when measured free VRAM is sufficient and end-to-end throughput improves; it is a separate manifest and never displaces required context/KV capacity on T4.
- Build future query plans/result previews on CPU where deterministic, but do not query early with stale `as_of`/revision data or retrieve a different pinned block before its declared epoch.

Record embedding queue/batch/time, Torch/CPU utilization, FTS/vector/SQLite time, RAM, and model-slot wait. Utilization smoke tests run the same index/query workload with T4 and L4 budgets and prove deterministic rankings, bounded queues/RAM, no writer starvation, and forward progress of model-ready episodes.

## Durable backup and restore

The active DB lives on local disk. Create a consistent snapshot with SQLite's online backup API after a bounded checkpoint, then atomically copy the single snapshot and a manifest to Drive. Do not independently copy `db`, `db-wal`, and `db-shm` as though they were one consistent file.

Manifest fields include namespace, schema/config/ranking/embedding versions, `memory_operations` revision, database SHA-256, source refs/catalog revision where applicable, and creation time. Use throttled background snapshots after mutations plus a forced snapshot at episode/sweep checkpoint and cleanup.

Restore:

1. Copy to a new local path atomically; never overwrite a newer local database blindly.
2. Run `PRAGMA integrity_check`, foreign-key checks, schema/config/revision/hash validation, and vector dimension/model checks.
3. Verify FTS and a canary vector query, then compare database mutation revision with the journal/checkpoint.
4. Restore missing newer operations only through the explicit run-recovery procedure in part 6, never through ordinary read-only replay.
5. Include tombstones so an older backup cannot resurrect forgotten content.

Drive failure leaves the local DB usable and records a durability lag; it is not silently reported as fully checkpointed.

## Retrieval metrics

Implement hand-testable:

- **Recall@5 and Recall@10:** a query hits when at least one currently valid evidence memory id from its gold set is returned; also report all-gold coverage when a question has multiple required facts.
- **NDCG@k:** graded evidence relevance when labels provide it, otherwise binary relevance, with deterministic id handling and zero-query denominators reported as null.
- **Supersession correctness:** old-only queries/audit can see history, ordinary search returns the new fact, and no active result contains both sides of a chain.
- **Deletion correctness:** no forgotten value appears in SQL, FTS, vector candidates, backup, or ordinary/audit search outside minimal tombstone metadata.
- **Time-filter correctness:** event time versus ingestion/validity time and questions with no explicit date.

Part 7 maps these to LongMemEval evidence labels. Fake-embedding tests verify arithmetic only, never a semantic recall claim.

## Tasks

1. Implement strict record/query values, namespace scoping, schema migration, deterministic ids, and transactional write/supersede/forget.
2. Add the FTS5 probe and parameterized lexical queries; reject unavailable real configurations explicitly.
3. Implement the local embedding protocol/provider and exact vector leg with identity/dimension checks.
4. Implement deterministic hybrid fusion, date rules, ranking configs, and canonical result blocks.
5. Implement online backup/restore, integrity validation, tombstones, revision tracking, and maintenance entry points.
6. Add metric functions and a small labeled cross-session fixture covering update, contradiction, deletion, and time-aware retrieval.
7. Add `tidemark-memory-gate` and `tidemark-memory-smoke` notebook cells. The real smoke pins/downloads the embedding model on CPU, checks its dimension, and stores/retrieves a fact across two fresh DB connections.
8. Implement the shared batched embedding/indexing service and resource-budgeted reader/writer queues. Autotune CPU threads/batches on T4 and L4; probe optional GPU embeddings only under spare-VRAM and end-to-end-throughput gates.
9. Run the phase-delivery protocol: commit the tested memory implementation, write the canonical Handoff for part 6, commit it, automatically push both commits, and verify remote `main`.

## Tests

- Identical writes are idempotent; changed writes create one valid supersession chain and concurrent predecessor updates fail rather than losing one.
- FTS/SQL/vector candidate sets never include another namespace, forgotten rows, or inactive superseded rows by default.
- User text containing FTS operators, quotes, wildcard-like strings, Unicode, or SQL punctuation is treated as query data.
- Ranking ties and missing one retrieval leg follow declared behavior with stable memory-id ordering.
- Date rules do not invent a range for an undated question; model rewriting is absent from the default path.
- Backup/restore detects torn/wrong-hash/wrong-revision/wrong-model files and preserves tombstones.
- A timed-out writer/backup cannot expose partial records or claim a durable revision that was not committed.
- Metric fixtures hand-check recall, NDCG, supersession, time filters, and zero denominators.
- No test needs a GPU or hosted embedding API; the real embedding smoke remains a separate notebook acceptance.

Run:

```bash
.venv/bin/python -m pytest tests/test_tidemark_memory_store.py tests/test_tidemark_memory_retrieval.py \
  tests/test_tidemark_embeddings.py tests/test_tidemark_memory_backup.py
.venv/bin/python -m pytest
```

## Phase delivery and handoff

After all storage/retrieval/backup tests, real embedding smoke, and the same memory workload on T4 and L4 pass, follow part 1's mandatory two-commit/one-push protocol automatically:

1. Stage only part 5's memory/store/retrieval/embedding code, sanitized fixtures, tests, notebook, README, and deliberate shared changes; never stage SQLite databases, vectors, embeddings, Drive snapshots, memory contents, or unrelated dirty paths.
2. Create the implementation commit `Tidemark: complete part 5 long-term memory`, update `Tidemark/Handoff.md`, then create `Tidemark: hand off part 5`.
3. Record the implementation SHA, SQLite/FTS/schema versions, embedding/preprocessing/ranking config and revision, recall/supersession/deletion evidence, resource settings, run ids, namespace/revision identifiers, and local/per-GPU Drive snapshot/restore commands.
4. Point the next session to part 6 and its first security/replay fixture. Push both commits to `origin/main` and verify remote `main` equals the Handoff commit before exposing memory tools.

A failed integrity/deletion/tombstone test, non-deterministic ranking, missing real embedding/two-GPU evidence, or failed publication blocks completion. Handoff contains paths and revisions only, never raw memory values or credentials.

## Acceptance

- A fact survives a fresh connection/session with source refs, timestamps, trust, and namespace intact.
- Hybrid retrieval is deterministic, current-only by default, and produces Recall@5/@10 and NDCG evidence on labeled fixtures.
- Supersession cannot leave two active contradictory values; forget removes every searchable copy and survives backup/restore.
- The real pinned embedding model is verified in the notebook before semantic retrieval results are reported.
- Active state is local and Drive copies are consistent, integrity-checked snapshots with revision evidence.
- The same memory workload runs on T4 and L4 with deterministic serial-equivalent rankings and measured embedding/indexing overlap; the selected CPU configuration stays within each runtime's vCPU/RAM budget.
- The tested implementation and canonical part 5 Handoff are committed and automatically pushed, remote `main` is verified, and part 6 can start in a fresh session with exact schema/model/revision/snapshot context.

## Pitfalls

- Copying a live WAL database to Drive as three unrelated files or overwriting a newer local DB during restore.
- Letting the model supply namespace, trust, embedding identity, arbitrary FTS syntax, or an unrestricted deletion query.
- Allowing old and new values to remain active and counting both as independent supporting evidence.
- Treating `created_at` as the fact's valid time or inventing date filters for undated questions.
- Assuming FTS5, pooling, embedding dimension, or vector-index availability instead of probing/pinning it.
- Retrieving different pinned memories on every render and blaming llama-server for the resulting prefix miss.
- Calling fake-hash similarity a semantic retrieval evaluation.
