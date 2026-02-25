"""Tests for confusion_set.py: word alignment, confusion span extraction,
uncertainty metrics, and gating."""

import numpy as np
import pytest

from confusionrag.confusion_set import (
    ConfusionSetResult,
    ConfusionSpan,
    _align_words,
    _disagreement_mass,
    _entropy,
    _margin,
    _softmax,
    build_confusion_sets,
)


# ---------------------------------------------------------------------------
# Word alignment
# ---------------------------------------------------------------------------

class TestAlignWords:
    def test_identical(self):
        alignment = _align_words(["a", "b", "c"], ["a", "b", "c"])
        ref_words = [r for r, _ in alignment]
        hyp_words = [h for _, h in alignment]
        assert ref_words == ["a", "b", "c"]
        assert hyp_words == ["a", "b", "c"]

    def test_substitution(self):
        alignment = _align_words(["the", "cat", "sat"], ["the", "dog", "sat"])
        subs = [(r, h) for r, h in alignment if r != h]
        assert len(subs) == 1
        assert subs[0] == ("cat", "dog")

    def test_insertion(self):
        alignment = _align_words(["a", "b"], ["a", "x", "b"])
        assert any(r is None for r, _ in alignment)

    def test_deletion(self):
        alignment = _align_words(["a", "b", "c"], ["a", "c"])
        assert any(h is None for _, h in alignment)

    def test_empty_ref(self):
        alignment = _align_words([], ["a", "b"])
        assert len(alignment) == 2
        assert all(r is None for r, _ in alignment)

    def test_empty_hyp(self):
        alignment = _align_words(["a", "b"], [])
        assert len(alignment) == 2
        assert all(h is None for _, h in alignment)


# ---------------------------------------------------------------------------
# Uncertainty metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_softmax_sums_to_one(self):
        w = _softmax(np.array([1.0, 2.0, 3.0]))
        assert abs(np.sum(w) - 1.0) < 1e-6

    def test_entropy_uniform(self):
        w = np.array([0.5, 0.5])
        ent = _entropy(w)
        assert ent > 0
        assert abs(ent - np.log(2)) < 1e-5

    def test_entropy_deterministic(self):
        w = np.array([1.0, 0.0])
        ent = _entropy(w)
        assert ent < 1e-6

    def test_margin_two_candidates(self):
        w = np.array([0.7, 0.3])
        assert abs(_margin(w) - 0.4) < 1e-6

    def test_margin_single_candidate(self):
        assert _margin(np.array([1.0])) == float("inf")

    def test_disagreement_mass(self):
        w = np.array([0.8, 0.1, 0.1])
        assert abs(_disagreement_mass(w) - 0.2) < 1e-6


# ---------------------------------------------------------------------------
# Confusion set construction
# ---------------------------------------------------------------------------

class TestBuildConfusionSets:
    def _make_nbest(self):
        return [
            ("the green library is open", -10.0, -5.0),
            ("the greene library is open", -11.0, -4.5),
            ("the cream library is open", -12.0, -4.0),
        ]

    def test_basic_output_type(self):
        result = build_confusion_sets(self._make_nbest())
        assert isinstance(result, ConfusionSetResult)
        assert result.top1_hypothesis == "the green library is open"

    def test_finds_disagreement_spans(self):
        result = build_confusion_sets(self._make_nbest(), gate_threshold=0.0)
        assert len(result.spans) > 0
        word_positions = {s.span_start for s in result.spans}
        assert 1 in word_positions  # position of green/greene/cream

    def test_candidates_include_all_variants(self):
        result = build_confusion_sets(self._make_nbest(), gate_threshold=0.0)
        span_at_1 = [s for s in result.spans if s.span_start == 1]
        assert len(span_at_1) == 1
        cands = set(span_at_1[0].candidates)
        assert "green" in cands
        assert "greene" in cands
        assert "cream" in cands

    def test_weights_sum_to_one(self):
        result = build_confusion_sets(self._make_nbest())
        for span in result.spans:
            assert abs(np.sum(span.weights) - 1.0) < 1e-5

    def test_entropy_gating(self):
        result = build_confusion_sets(
            self._make_nbest(), gate_metric="entropy", gate_threshold=0.01
        )
        assert any(result.is_uncertain)

    def test_high_threshold_blocks_gating(self):
        result = build_confusion_sets(
            self._make_nbest(), gate_metric="entropy", gate_threshold=100.0
        )
        assert not any(result.is_uncertain)

    def test_margin_gating(self):
        result = build_confusion_sets(
            self._make_nbest(), gate_metric="margin", gate_threshold=0.99
        )
        assert any(result.is_uncertain)

    def test_disagreement_mass_gating(self):
        result = build_confusion_sets(
            self._make_nbest(),
            gate_metric="disagreement_mass",
            gate_threshold=0.01,
        )
        assert any(result.is_uncertain)

    def test_empty_nbest(self):
        result = build_confusion_sets([])
        assert result.top1_hypothesis == ""
        assert len(result.spans) == 0

    def test_single_hypothesis(self):
        result = build_confusion_sets(
            [("hello world", -5.0, -2.0)], min_nbest=2
        )
        assert result.top1_hypothesis == "hello world"
        assert len(result.spans) == 0

    def test_identical_hypotheses(self):
        nbest = [
            ("same sentence", -5.0, -2.0),
            ("same sentence", -6.0, -3.0),
        ]
        result = build_confusion_sets(nbest)
        assert len(result.spans) == 0

    def test_span_to_dict(self):
        result = build_confusion_sets(self._make_nbest())
        if result.spans:
            d = result.spans[0].to_dict()
            assert "span_start" in d
            assert "candidates" in d
            assert isinstance(d["weights"], list)
