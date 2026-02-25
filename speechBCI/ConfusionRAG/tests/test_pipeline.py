"""Integration tests for pipeline.py."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from confusionrag.config import ConfusionRAGConfig
from confusionrag.pipeline import decode_with_confusion_rag
from confusionrag.retriever import Retriever
from confusionrag.tracer import RunTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nbest_outputs(n_sentences=5, n_best=10):
    """Synthetic N-best lists with controlled disagreement."""
    base_words = ["the", "quick", "brown", "fox", "jumps"]
    alternatives = {
        1: ["slow", "thick"],      # position 1
        2: ["red", "green"],       # position 2
    }

    nbest_outputs = []
    for _ in range(n_sentences):
        nbest = []
        nbest.append((" ".join(base_words), -10.0, -5.0))
        for k in range(1, n_best):
            words = list(base_words)
            for pos, alts in alternatives.items():
                if np.random.random() > 0.5:
                    words[pos] = np.random.choice(alts)
            nbest.append((" ".join(words), -10.0 - k, -5.0 - k * 0.5))
        nbest_outputs.append(nbest)
    return nbest_outputs


def _make_inference_out(n_sentences=5):
    """Minimal inference_out dict with transcriptions."""
    transcriptions = []
    text = "the quick brown fox jumps"
    for _ in range(n_sentences):
        encoded = np.array([ord(c) for c in text] + [0] * 10, dtype=np.int32)
        transcriptions.append(encoded)
    return {"transcriptions": np.array(transcriptions)}


def _mock_rescore(scores):
    def _side_effect(model, tokenizer, texts, length_penalty=0.0):
        return [0.0] * len(texts)
    return patch(
        "confusionrag.constrained_llm._rescore_with_llm",
        side_effect=_side_effect,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_basic_output_structure(self):
        nbest = _make_nbest_outputs(3)
        inf_out = _make_inference_out(3)
        cfg = ConfusionRAGConfig(
            llm_mode="nbest_rescore",
            gate_threshold=0.0,
            trace_enabled=False,
        )

        with _mock_rescore(None):
            result = decode_with_confusion_rag(
                nbest, inf_out, None, None, retriever=None, config=cfg,
            )

        assert "decoded_transcripts" in result
        assert "confidences" in result
        assert "trace" in result
        assert len(result["decoded_transcripts"]) == 3
        assert len(result["confidences"]) == 3

    def test_with_retriever(self):
        corpus = [
            "the quick brown fox jumps",
            "a slow red fox sits",
            "the green library opens",
        ]
        retriever = Retriever(corpus, top_k=2)
        nbest = _make_nbest_outputs(2)
        inf_out = _make_inference_out(2)
        cfg = ConfusionRAGConfig(
            llm_mode="nbest_rescore",
            gate_threshold=0.0,
            trace_enabled=False,
        )

        with _mock_rescore(None):
            result = decode_with_confusion_rag(
                nbest, inf_out, None, None, retriever, cfg,
            )

        assert len(result["decoded_transcripts"]) == 2

    def test_span_choice_mode(self):
        nbest = _make_nbest_outputs(2)
        inf_out = _make_inference_out(2)
        retriever = Retriever(
            ["the quick brown fox jumps", "the thick green fox jumps"], top_k=2
        )
        cfg = ConfusionRAGConfig(
            llm_mode="span_choice",
            gate_threshold=0.0,
            trace_enabled=False,
        )

        with _mock_rescore(None):
            result = decode_with_confusion_rag(
                nbest, inf_out, None, None, retriever=retriever, config=cfg,
            )

        for st in result["trace"].sentences:
            if st.n_uncertain_spans > 0:
                assert st.decision_mode == "span_choice"

    def test_nbest_rescore_change_was_correct_true(self):
        nbest = [[
            ("the dog sat", -5.0, -3.0),
            ("the cat sat", -6.0, -4.0),
        ]]
        inf_out = {
            "transcriptions": np.array(
                [[ord(c) for c in "the cat sat"] + [0] * 10], dtype=np.int32
            )
        }
        cfg = ConfusionRAGConfig(
            llm_mode="nbest_rescore",
            gate_threshold=0.0,
            llm_alpha=1.0,
            acoustic_scale=0.0,
            trace_enabled=False,
        )

        with patch(
            "confusionrag.constrained_llm._rescore_with_llm",
            return_value=[0.0, 10.0],
        ):
            result = decode_with_confusion_rag(
                nbest, inf_out, None, None, retriever=None, config=cfg,
            )

        st = result["trace"].sentences[0]
        decision_span = next(
            sp for sp in st.spans
            if sp.llm_decision and sp.llm_decision.get("mode") == "nbest_rescore"
        )
        assert decision_span.llm_decision["changed_from_top1"] is True
        assert decision_span.llm_decision["selected"] == "the cat sat"
        assert decision_span.llm_decision["change_was_correct"] is True

    def test_nbest_rescore_change_was_correct_false(self):
        nbest = [[
            ("the cat sat", -5.0, -3.0),
            ("the dog sat", -6.0, -4.0),
        ]]
        inf_out = {
            "transcriptions": np.array(
                [[ord(c) for c in "the cat sat"] + [0] * 10], dtype=np.int32
            )
        }
        cfg = ConfusionRAGConfig(
            llm_mode="nbest_rescore",
            gate_threshold=0.0,
            llm_alpha=1.0,
            acoustic_scale=0.0,
            trace_enabled=False,
        )

        with patch(
            "confusionrag.constrained_llm._rescore_with_llm",
            return_value=[0.0, 10.0],
        ):
            result = decode_with_confusion_rag(
                nbest, inf_out, None, None, retriever=None, config=cfg,
            )

        st = result["trace"].sentences[0]
        decision_span = next(
            sp for sp in st.spans
            if sp.llm_decision and sp.llm_decision.get("mode") == "nbest_rescore"
        )
        assert decision_span.llm_decision["changed_from_top1"] is True
        assert decision_span.llm_decision["selected"] == "the dog sat"
        assert decision_span.llm_decision["change_was_correct"] is False

    def test_trace_file_written(self):
        nbest = _make_nbest_outputs(2)
        inf_out = _make_inference_out(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ConfusionRAGConfig(
                llm_mode="nbest_rescore",
                gate_threshold=0.0,
                trace_enabled=True,
                trace_dir=tmpdir,
            )

            with _mock_rescore(None):
                result = decode_with_confusion_rag(
                    nbest, inf_out, None, None, retriever=None, config=cfg,
                )

            assert result["trace_path"] is not None
            assert os.path.exists(result["trace_path"])

            loaded = RunTrace.load(result["trace_path"])
            assert loaded.run_id == result["trace"].run_id

    def test_high_threshold_keeps_top1(self):
        nbest = _make_nbest_outputs(3)
        inf_out = _make_inference_out(3)
        cfg = ConfusionRAGConfig(
            gate_threshold=999.0,
            trace_enabled=False,
        )

        with _mock_rescore(None):
            result = decode_with_confusion_rag(
                nbest, inf_out, None, None, retriever=None, config=cfg,
            )

        for st in result["trace"].sentences:
            assert st.decision_mode == "kept_top1"

    def test_session_memory_grows(self):
        corpus = ["some training sentence"]
        retriever = Retriever(corpus, top_k=1, session_memory_size=5)
        nbest = _make_nbest_outputs(3)
        inf_out = _make_inference_out(3)
        cfg = ConfusionRAGConfig(
            gate_threshold=0.0,
            trace_enabled=False,
        )

        with _mock_rescore(None):
            decode_with_confusion_rag(
                nbest, inf_out, None, None, retriever, cfg,
            )

        assert len(retriever.get_session_memory()) == 3

    def test_empty_nbest(self):
        nbest = [[], [], []]
        inf_out = _make_inference_out(3)
        cfg = ConfusionRAGConfig(trace_enabled=False)

        with _mock_rescore(None):
            result = decode_with_confusion_rag(
                nbest, inf_out, None, None, retriever=None, config=cfg,
            )

        assert all(d == "" for d in result["decoded_transcripts"])
