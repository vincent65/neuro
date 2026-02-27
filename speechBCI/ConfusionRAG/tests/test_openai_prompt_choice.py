"""Tests for OpenAI prompt-choice helper with mocked API responses."""

from __future__ import annotations

import types

from benchmarks.openai_prompt_choice import choose_candidate_with_openai


class _FakeResponses:
    def __init__(self, output_text: str):
        self._output_text = output_text

    def create(self, **kwargs):
        _ = kwargs
        return types.SimpleNamespace(output_text=self._output_text)


class _FakeOpenAI:
    def __init__(self, api_key=None, timeout=None):
        _ = (api_key, timeout)
        self.responses = _FakeResponses('{"chosen_candidate":"beta","confidence":0.73}')


class _FakeOpenAIInvalid:
    def __init__(self, api_key=None, timeout=None):
        _ = (api_key, timeout)
        self.responses = _FakeResponses('{"chosen_candidate":"not_in_set"}')


class _FakeOpenAIFreeText:
    def __init__(self, api_key=None, timeout=None):
        _ = (api_key, timeout)
        self.responses = _FakeResponses("I choose BETA because it best matches context.")


class _FakeOpenAICodeFenceJson:
    def __init__(self, api_key=None, timeout=None):
        _ = (api_key, timeout)
        self.responses = _FakeResponses("```json\n{\"chosen_candidate\":\"beta\"}\n```")


def test_missing_api_key_falls_back(monkeypatch):
    fake_module = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = choose_candidate_with_openai(
        masked_sentence="this is [MASK] test",
        candidates=["alpha", "beta"],
        candidate_weights=[0.9, 0.1],
    )
    assert result.used_fallback is True
    assert result.fallback_reason == "missing_api_key"
    assert result.chosen_candidate == "alpha"


def test_valid_response_uses_model_choice(monkeypatch):
    fake_module = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = choose_candidate_with_openai(
        masked_sentence="this is [MASK] test",
        candidates=["alpha", "beta"],
        candidate_weights=[0.9, 0.1],
    )
    assert result.used_fallback is False
    assert result.chosen_candidate == "beta"
    assert result.confidence == 0.73


def test_out_of_set_response_falls_back(monkeypatch):
    fake_module = types.SimpleNamespace(OpenAI=_FakeOpenAIInvalid)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = choose_candidate_with_openai(
        masked_sentence="this is [MASK] test",
        candidates=["alpha", "beta"],
        candidate_weights=[0.9, 0.1],
        max_retries=1,
    )
    assert result.used_fallback is True
    assert result.fallback_reason == "invalid_or_failed_response"
    assert result.chosen_candidate == "alpha"


def test_free_text_candidate_mention_is_accepted(monkeypatch):
    fake_module = types.SimpleNamespace(OpenAI=_FakeOpenAIFreeText)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = choose_candidate_with_openai(
        masked_sentence="this is [MASK] test",
        candidates=["alpha", "beta"],
        candidate_weights=[0.9, 0.1],
    )
    assert result.used_fallback is False
    assert result.chosen_candidate == "beta"


def test_json_code_fence_is_parsed(monkeypatch):
    fake_module = types.SimpleNamespace(OpenAI=_FakeOpenAICodeFenceJson)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = choose_candidate_with_openai(
        masked_sentence="this is [MASK] test",
        candidates=["alpha", "beta"],
        candidate_weights=[0.9, 0.1],
    )
    assert result.used_fallback is False
    assert result.chosen_candidate == "beta"
