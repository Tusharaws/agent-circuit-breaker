"""Tests for SLMClient (Phase 4). Written before the implementation
(evaluator.slm does not exist yet) -- run `pytest` to see them fail with
a collection error until src/evaluator/slm.py exists.

Unit tests inject a fake tokenizer + fake generate_fn -- no real model
load, fast and network-independent. A separate, clearly-marked
integration test loads the REAL mlx-community Qwen2.5-0.5B-Instruct
model (already downloaded/cached locally) and proves the actual stack
works, not just the wrapper's plumbing.
"""
import time

import pytest

from evaluator.slm import DEFAULT_MODEL_NAME, SLMClient


class FakeTokenizer:
    def __init__(self):
        self.last_messages = None

    def apply_chat_template(self, messages, add_generation_prompt=True):
        self.last_messages = messages
        return f"FORMATTED[{messages[0]['content']}]"


class FakeModel:
    pass


def _fake_generate_fn(model, tokenizer, prompt, max_tokens, verbose=False):
    return f"fake response to: {prompt} (max_tokens={max_tokens})"


@pytest.fixture
def client():
    return SLMClient(model=FakeModel(), tokenizer=FakeTokenizer(), generate_fn=_fake_generate_fn)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_requires_either_model_name_or_injected_model_and_tokenizer():
    # injected model+tokenizer means model_name is never consulted -- must not raise
    client = SLMClient(model=FakeModel(), tokenizer=FakeTokenizer(), generate_fn=_fake_generate_fn)
    assert client is not None


# ---------------------------------------------------------------------------
# generate(): call shaping
# ---------------------------------------------------------------------------


def test_generate_formats_prompt_via_chat_template(client):
    result = client.generate("classify this window")

    assert "FORMATTED[classify this window]" in result


def test_generate_passes_max_tokens_through(client):
    result = client.generate("hello", max_tokens=42)

    assert "max_tokens=42" in result


def test_generate_default_max_tokens_is_200(client):
    result = client.generate("hello")

    assert "max_tokens=200" in result


def test_generate_returns_the_generate_fn_result_directly(client):
    result = client.generate("test prompt")

    assert result == "fake response to: FORMATTED[test prompt] (max_tokens=200)"


# ---------------------------------------------------------------------------
# Integration: the real model (already downloaded/cached)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_model_produces_non_empty_generation():
    client = SLMClient(model_name=DEFAULT_MODEL_NAME)

    start = time.perf_counter()
    result = client.generate("Reply with the single word: ok", max_tokens=10)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"\nreal SLM call: {elapsed_ms:.0f}ms, response={result!r}")

    assert isinstance(result, str)
    assert len(result.strip()) > 0
