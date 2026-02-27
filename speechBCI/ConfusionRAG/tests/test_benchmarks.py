"""Tests for benchmark scripts."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.run_baselines import baseline_always_on_rag, baseline_context_only


def _tiny_nbest():
    return [
        [
            ("alpha one", -1.0, -1.0),
            ("alpha two", -2.0, -2.0),
        ],
        [
            ("beta one", -1.0, -1.0),
            ("beta two", -2.0, -2.0),
        ],
    ]


def _tiny_inference_out():
    texts = ["alpha two", "beta two"]
    max_len = max(len(t) for t in texts) + 10
    arr = []
    for text in texts:
        encoded = [ord(c) for c in text]
        encoded = encoded + [0] * (max_len - len(encoded))
        arr.append(encoded)
    return {"transcriptions": np.array(arr, dtype=np.int32)}


class TestRunBaselines:
    def test_always_on_rag_requires_corpus(self):
        with pytest.raises(ValueError, match="requires a non-empty corpus_path"):
            baseline_always_on_rag(
                _tiny_nbest(),
                _tiny_inference_out(),
                ["alpha two", "beta two"],
                llm=None,
                tok=None,
                corpus=[],
            )

    def test_context_only_uses_session_memory_as_evidence(self):
        nbest = _tiny_nbest()
        inf_out = _tiny_inference_out()
        reference = ["alpha two", "beta two"]

        captured_text_batches = []

        def _score_with_context(_model, _tok, texts, length_penalty=0.0, max_batch_size=0):
            captured_text_batches.append(texts)
            if len(captured_text_batches) == 1:
                # First sentence: choose second hypothesis ("alpha two").
                return [0.0, 10.0]
            # Second sentence: ensure context includes prior decoded sentence.
            assert "Context:" in texts[0]
            assert "alpha two" in texts[0]
            # Choose second hypothesis again ("beta two").
            return [0.0, 10.0]

        with patch(
            "confusionrag.constrained_llm._rescore_with_llm",
            side_effect=_score_with_context,
        ):
            result = baseline_context_only(nbest, inf_out, reference, llm=None, tok=None)

        assert result["name"] == "context_only"
        assert len(captured_text_batches) == 2
