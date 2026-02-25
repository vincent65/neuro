"""Tests for constrained_llm.py: span-choice and N-best rescoring with mocked LLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import List

import numpy as np
import pytest

from confusionrag.confusion_set import ConfusionSetResult, ConfusionSpan, build_confusion_sets
from confusionrag.constrained_llm import (
    _build_evidence_prefix,
    _rescore_with_llm,
    nbest_rescore_decode,
    span_choice_decode,
)
from confusionrag.retriever import RetrievalResult
from confusionrag.tracer import SentenceTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_rescore(scores: List[float]):
    """Return a patch that replaces _rescore_with_llm with fixed scores."""
    return patch(
        "confusionrag.constrained_llm._rescore_with_llm",
        return_value=scores,
    )


def _make_confusion_result():
    """3 hypotheses differing at position 1 (green/greene/cream)."""
    return build_confusion_sets([
        ("the green library is open", -10.0, -5.0),
        ("the greene library is open", -11.0, -4.5),
        ("the cream library is open", -12.0, -4.0),
    ], gate_metric="entropy", gate_threshold=0.0)


def _make_retrieval_results(confusion_result: ConfusionSetResult):
    rr = {}
    for idx, (span, unc) in enumerate(
        zip(confusion_result.spans, confusion_result.is_uncertain)
    ):
        if unc:
            rr[idx] = RetrievalResult(
                query="test query",
                retrieved_docs=["the green library is a great place"],
                scores=[5.0],
                retrieval_time_ms=1.0,
            )
    return rr


# ---------------------------------------------------------------------------
# Evidence prefix
# ---------------------------------------------------------------------------

class TestBuildEvidencePrefix:
    def test_empty(self):
        assert _build_evidence_prefix([]) == ""

    def test_formats_docs(self):
        prefix = _build_evidence_prefix(["doc1", "doc2"])
        assert "Context:" in prefix
        assert "doc1" in prefix
        assert "doc2" in prefix

    def test_max_docs(self):
        prefix = _build_evidence_prefix(
            [f"doc{i}" for i in range(10)], max_docs=2
        )
        assert "doc0" in prefix
        assert "doc1" in prefix
        assert "doc5" not in prefix


# ---------------------------------------------------------------------------
# Span-choice mode
# ---------------------------------------------------------------------------

class TestSpanChoice:
    def test_selects_best_candidate(self):
        cs = _make_confusion_result()
        rr = _make_retrieval_results(cs)
        st = SentenceTrace()

        n_candidates = len(cs.spans[0].candidates) if cs.spans else 3
        mock_scores = [0.0] * n_candidates
        mock_scores[0] = 10.0  # first candidate wins

        with _mock_rescore(mock_scores):
            decoded = span_choice_decode(
                None, None, cs, rr, st, length_penalty=0.0
            )

        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_trace_records_mode(self):
        cs = _make_confusion_result()
        rr = _make_retrieval_results(cs)
        st = SentenceTrace()

        n_candidates = len(cs.spans[0].candidates) if cs.spans else 3
        with _mock_rescore([1.0] * n_candidates):
            span_choice_decode(None, None, cs, rr, st)

        assert st.decision_mode == "span_choice"
        for sp in st.spans:
            if sp.llm_decision is not None:
                assert sp.llm_decision["mode"] == "span_choice"

    def test_output_from_confusion_set_only(self):
        cs = _make_confusion_result()
        rr = _make_retrieval_results(cs)
        st = SentenceTrace()

        n_candidates = len(cs.spans[0].candidates) if cs.spans else 3
        with _mock_rescore([1.0] * n_candidates):
            decoded = span_choice_decode(None, None, cs, rr, st)

        decoded_words = set(decoded.split())
        all_candidates = set()
        for span in cs.spans:
            all_candidates.update(span.candidates)
        all_original = set(cs.top1_hypothesis.split())
        allowed = all_candidates | all_original
        assert decoded_words.issubset(allowed)

    def test_changed_from_top1_flag(self):
        cs = _make_confusion_result()
        rr = _make_retrieval_results(cs)
        st = SentenceTrace()

        if cs.spans:
            n_candidates = len(cs.spans[0].candidates)
            # Make the last candidate win
            scores = [0.0] * n_candidates
            scores[-1] = 10.0
            with _mock_rescore(scores):
                span_choice_decode(None, None, cs, rr, st)

            changed_spans = [
                sp for sp in st.spans
                if sp.llm_decision and sp.llm_decision.get("changed_from_top1")
            ]
            assert len(changed_spans) > 0


# ---------------------------------------------------------------------------
# N-best rescoring mode
# ---------------------------------------------------------------------------

class TestNbestRescore:
    def test_selects_best_hypothesis(self):
        cs = _make_confusion_result()
        rr = _make_retrieval_results(cs)
        st = SentenceTrace()

        n_hyps = len(cs.nbest_hypotheses)
        mock_scores = [0.0] * n_hyps
        mock_scores[1] = 100.0  # second hypothesis wins

        with _mock_rescore(mock_scores):
            decoded = nbest_rescore_decode(
                None, None, cs, rr, st,
                alpha=1.0, acoustic_scale=0.0, length_penalty=0.0,
            )

        assert decoded == cs.nbest_hypotheses[1]

    def test_trace_records_mode(self):
        cs = _make_confusion_result()
        rr = _make_retrieval_results(cs)
        st = SentenceTrace()

        n_hyps = len(cs.nbest_hypotheses)
        with _mock_rescore([0.0] * n_hyps):
            nbest_rescore_decode(None, None, cs, rr, st)

        assert st.decision_mode == "nbest_rescore"
        nbest_spans = [
            sp for sp in st.spans
            if sp.llm_decision and sp.llm_decision.get("mode") == "nbest_rescore"
        ]
        assert len(nbest_spans) > 0

    def test_combined_scoring(self):
        cs = _make_confusion_result()
        rr = {}
        st = SentenceTrace()

        n_hyps = len(cs.nbest_hypotheses)
        # All LLM scores zero, so old scores dominate
        with _mock_rescore([0.0] * n_hyps):
            decoded = nbest_rescore_decode(
                None, None, cs, rr, st,
                alpha=0.0, acoustic_scale=1.0,
            )

        # With alpha=0, acoustic + old LM scores decide
        assert decoded in cs.nbest_hypotheses

    def test_empty_hypotheses(self):
        cs = ConfusionSetResult(
            top1_hypothesis="",
            nbest_hypotheses=[],
            nbest_scores=[],
        )
        st = SentenceTrace()
        decoded = nbest_rescore_decode(None, None, cs, {}, st)
        assert decoded == ""

    def test_candidate_scores_in_trace(self):
        cs = _make_confusion_result()
        rr = _make_retrieval_results(cs)
        st = SentenceTrace()

        n_hyps = len(cs.nbest_hypotheses)
        with _mock_rescore([1.0] * n_hyps):
            nbest_rescore_decode(None, None, cs, rr, st)

        nbest_span = next(
            (sp for sp in st.spans
             if sp.llm_decision and sp.llm_decision.get("mode") == "nbest_rescore"),
            None,
        )
        assert nbest_span is not None
        scores = nbest_span.llm_decision["candidate_scores"]
        assert len(scores) == n_hyps
        for entry in scores:
            assert "llm_score" in entry
            assert "old_lm_score" in entry
            assert "acoustic_score" in entry
            assert "combined_score" in entry
