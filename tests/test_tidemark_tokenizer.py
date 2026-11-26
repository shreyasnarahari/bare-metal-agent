"""The tokenizer protocol, the parity probe, and caching only after parity passes."""

import json
from pathlib import Path

import httpx
import pytest

from tidemark.tokenizer import (
    INVALID_CASES,
    PLAIN_CASES,
    CachedTokenizer,
    HFTokenizer,
    InvalidText,
    ParityCase,
    ParityError,
    RECITATION_TAIL,
    RegexTokenizer,
    ServerTokenizer,
    check_text,
    main,
    parity_corpus,
    run_parity,
    token_ids_sha256,
)

from tidemark_support import qwen_like_render

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "tidemark" / "parity_cases.json").read_text(encoding="utf-8"))


def render(messages, tools):
    return qwen_like_render(messages, tools)


class SplitMarkers(RegexTokenizer):
    """A 'server' that splits <|im_start|> into pieces: a markup-boundary disagreement."""

    def encode(self, text):
        return super().encode(text.replace("<|", "< |"))


class OffByOne(RegexTokenizer):
    """A 'server' whose ids differ from the local tokenizer for one piece only."""

    def encode(self, text):
        ids = list(super().encode(text))
        return tuple(i + 1 if index == 3 else i for index, i in enumerate(ids))


def test_the_fake_tokenizer_is_deterministic_and_matches_its_fixture():
    tokenizer = RegexTokenizer()
    for case in PLAIN_CASES:
        expected = FIXTURE["cases"][case.case_id]
        ids = tokenizer.encode(case.text)
        assert len(ids) == expected["count"] and token_ids_sha256(ids) == expected["ids_sha256"], case.case_id
    assert tokenizer.encode("") == ()
    assert tokenizer.identity()["kind"] == "fake"


def test_the_corpus_covers_unicode_markup_and_whitespace():
    texts = {case.case_id: case.text for case in PLAIN_CASES}
    assert "́" in texts["combining"] and "👨‍👩‍👧" in texts["emoji"] and "東京" in texts["non_ascii"]
    assert "\r\n" in texts["newlines"] and texts["markup_adjacent"].startswith("<|im_start|>")
    assert texts["tool_markup_adjacent"].startswith("x<tool_call>") and texts["tool_markup_adjacent"].endswith("y")


@pytest.mark.parametrize("case", INVALID_CASES, ids=lambda case: case.case_id)
def test_lone_surrogates_are_rejected_before_any_tokenizer(case):
    with pytest.raises(InvalidText):
        check_text(case.text)
    with pytest.raises(InvalidText):
        RegexTokenizer().encode(case.text)


def test_the_rendered_corpus_covers_system_tools_generation_marker_and_recitation():
    cases = {case.case_id: case.text for case in parity_corpus(render)}
    assert cases["rendered_empty_system"].startswith("<|im_start|>system\n<|im_end|>")
    assert "# Tools" in cases["rendered_tools"] and "# Tools" not in cases["rendered_system"]
    assert all(cases[name].endswith("<|im_start|>assistant\n") for name in cases if name.startswith("rendered_"))
    assert RECITATION_TAIL.strip() in cases["rendered_after_recitation"]
    assert cases["rendered_before_recitation"] != cases["rendered_after_recitation"]
    assert len(parity_corpus()) == len(PLAIN_CASES)


def test_identical_tokenizers_pass_parity():
    report = run_parity(RegexTokenizer(), RegexTokenizer(), parity_corpus(render), identity={"template_sha256": "t1"})
    assert report.passed and report.mismatches == []
    assert all(report.invalid_rejected.values()) and report.identity == {"template_sha256": "t1"}
    payload = json.dumps(report.as_dict())
    assert "ord_1001" not in payload  # counts and hashes only, never the prompts


def test_a_markup_boundary_disagreement_fails_parity():
    report = run_parity(RegexTokenizer(), SplitMarkers(), parity_corpus(render))
    assert not report.passed
    assert "markup_adjacent" in report.mismatches and "rendered_tools" in report.mismatches
    assert "ascii" not in report.mismatches


def test_mismatched_token_ids_name_the_first_differing_position():
    report = run_parity(RegexTokenizer(), OffByOne(), [PLAIN_CASES[1]])
    (result,) = report.results
    assert not result.match and result.first_mismatch == 3 and result.local_count == result.server_count


def test_a_length_difference_is_a_mismatch_at_the_shorter_length():
    class Shorter(RegexTokenizer):
        def encode(self, text):
            return super().encode(text)[:-1]

    (result,) = run_parity(RegexTokenizer(), Shorter(), [ParityCase("x", "a b c")]).results
    assert result.first_mismatch == 4 and not result.match


def test_a_template_revision_changes_the_rendered_case_hashes():
    def revised(messages, tools):
        return render(messages, tools).replace("<|im_start|>assistant\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n")

    before = {r.case_id: r.text_sha256 for r in run_parity(RegexTokenizer(), RegexTokenizer(), parity_corpus(render)).results}
    after = {r.case_id: r.text_sha256 for r in run_parity(RegexTokenizer(), RegexTokenizer(), parity_corpus(revised)).results}
    assert before["ascii"] == after["ascii"] and before["rendered_tools"] != after["rendered_tools"]


def test_a_tokenizer_that_accepts_surrogates_fails_the_invalid_check(monkeypatch):
    import tidemark.tokenizer as module

    monkeypatch.setattr(module, "check_text", lambda text: text)
    report = module.run_parity(RegexTokenizer(), RegexTokenizer(), [PLAIN_CASES[1]])
    assert not report.passed and any("not rejected" in item for item in report.mismatches)


def test_counts_are_cached_only_after_parity_passes():
    with pytest.raises(ParityError):
        CachedTokenizer(RegexTokenizer(), None)
    failed = run_parity(RegexTokenizer(), SplitMarkers(), parity_corpus(render))
    with pytest.raises(ParityError):
        CachedTokenizer(RegexTokenizer(), failed)
    passed = run_parity(RegexTokenizer(), RegexTokenizer(), parity_corpus(render))
    cached = CachedTokenizer(RegexTokenizer(), passed, max_entries=2)
    assert cached.encode("one two") == RegexTokenizer().encode("one two")
    cached.encode("one two")
    assert (cached.hits, cached.misses) == (1, 1)
    cached.encode("a")
    cached.encode("b")
    cached.encode("one two")  # evicted by the two newer entries
    assert cached.misses == 4
    assert cached.identity()["parity"] == "passed" and cached.identity()["parity_sha256"] == passed.sha256()


def test_the_hf_adapter_encodes_without_special_tokens_and_records_hashes():
    class FakeHF:
        name_or_path = "Qwen/Qwen3-8B"
        chat_template = "{{ messages }}"

        class backend_tokenizer:
            @staticmethod
            def to_str():
                return '{"model": "bpe"}'

        def __init__(self):
            self.calls = []

        def encode(self, text, add_special_tokens=True):
            self.calls.append(add_special_tokens)
            return [ord(c) for c in text]

        def __len__(self):
            return 151_000

    fake = FakeHF()
    tokenizer = HFTokenizer(fake, revision="abc123")
    assert tokenizer.encode("hi") == (104, 105) and fake.calls == [False]
    identity = tokenizer.identity()
    assert identity["name"] == "Qwen/Qwen3-8B" and identity["revision"] == "abc123" and identity["vocab_size"] == 151_000
    assert len(identity["tokenizer_sha256"]) == 64 and len(identity["template_sha256"]) == 64


def test_the_server_adapter_asks_tokenize_like_completion_does():
    seen = []

    def handler(request):
        seen.append((request.headers.get("authorization"), json.loads(request.content)))
        return httpx.Response(200, json={"tokens": [1, 2, 3]})

    server = ServerTokenizer("http://127.0.0.1:9", "secret-key", transport=httpx.MockTransport(handler))
    assert server.encode("abc") == (1, 2, 3)
    assert seen == [("Bearer secret-key", {"content": "abc", "add_special": True})]
    assert "secret" not in json.dumps(server.identity())
    server.close()

    bad = ServerTokenizer("http://127.0.0.1:9", "k", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"tokens": ["x"]})))
    with pytest.raises(ValueError):
        bad.encode("abc")


def test_the_cli_requires_the_key_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCAL_KEY", raising=False)
    with pytest.raises(SystemExit):
        main(["--base-url", "http://127.0.0.1:9", "--model", "Qwen/Qwen3-8B", "--out", str(tmp_path / "p.json")])
