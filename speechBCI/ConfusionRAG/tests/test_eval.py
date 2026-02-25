"""Tests for eval.py: WER/CER, oracle metrics, slices, faithfulness audit."""

import numpy as np
import pytest

from confusionrag.eval import (
    build_word_freq,
    compute_cer,
    compute_oracle_wer,
    compute_slice_wer,
    compute_wer,
    compute_wer_with_ci,
    faithfulness_audit,
    high_uncertainty_mask,
    levenshtein,
    mode_breakdown_wer,
    oracle_gap_closed,
    rare_word_mask,
)
from confusionrag.tracer import RunTrace, SentenceTrace, SpanTrace


# ---------------------------------------------------------------------------
# Levenshtein / WER / CER
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_identical(self):
        assert levenshtein(["a", "b"], ["a", "b"]) == 0

    def test_one_substitution(self):
        assert levenshtein(["a", "b"], ["a", "c"]) == 1

    def test_insertion(self):
        assert levenshtein(["a", "b"], ["a", "x", "b"]) == 1

    def test_deletion(self):
        assert levenshtein(["a", "b", "c"], ["a", "c"]) == 1

    def test_empty(self):
        assert levenshtein([], ["a", "b"]) == 2
        assert levenshtein(["a"], []) == 1


class TestWER:
    def test_perfect(self):
        assert compute_wer(["hello world"], ["hello world"]) == 0.0

    def test_all_wrong(self):
        wer = compute_wer(["a b c"], ["x y z"])
        assert wer == 1.0

    def test_partial(self):
        wer = compute_wer(["the cat sat"], ["the dog sat"])
        assert abs(wer - 1.0 / 3.0) < 1e-6


class TestCER:
    def test_perfect(self):
        assert compute_cer(["abc"], ["abc"]) == 0.0

    def test_one_char_diff(self):
        cer = compute_cer(["abc"], ["axc"])
        assert abs(cer - 1.0 / 3.0) < 1e-6


class TestWERWithCI:
    def test_returns_three_values(self):
        wer, lo, hi = compute_wer_with_ci(
            ["the cat sat", "a b"], ["the dog sat", "a b"]
        )
        assert lo <= wer <= hi


# ---------------------------------------------------------------------------
# Oracle WER
# ---------------------------------------------------------------------------

class TestOracleWER:
    def test_oracle_picks_best(self):
        nbest = [
            [("the dog sat", -5.0, -2.0), ("the cat sat", -6.0, -3.0)],
        ]
        reference = ["the cat sat"]
        oracle = compute_oracle_wer(nbest, reference)
        assert oracle == 0.0

    def test_oracle_worse_than_all(self):
        nbest = [
            [("a b c", -5.0, -2.0), ("x y z", -6.0, -3.0)],
        ]
        reference = ["p q r"]
        oracle = compute_oracle_wer(nbest, reference)
        assert oracle == 1.0


class TestOracleGapClosed:
    def test_perfect_recovery(self):
        assert oracle_gap_closed(0.3, 0.1, 0.1) == 1.0

    def test_no_improvement(self):
        assert oracle_gap_closed(0.3, 0.3, 0.1) == 0.0

    def test_partial(self):
        gap = oracle_gap_closed(0.4, 0.3, 0.2)
        assert abs(gap - 0.5) < 1e-6

    def test_zero_gap(self):
        assert oracle_gap_closed(0.2, 0.2, 0.2) == 0.0


# ---------------------------------------------------------------------------
# Slice metrics
# ---------------------------------------------------------------------------

class TestSliceMetrics:
    def test_slice_wer(self):
        decoded = ["a b", "x y", "c d"]
        ref = ["a b", "a b", "c d"]
        mask = [True, True, False]
        wer = compute_slice_wer(decoded, ref, mask)
        assert wer is not None
        assert wer > 0

    def test_empty_mask(self):
        assert compute_slice_wer(["a"], ["b"], [False]) is None

    def test_high_uncertainty_mask(self):
        trace = RunTrace()
        trace.sentences = [
            SentenceTrace(n_uncertain_spans=2),
            SentenceTrace(n_uncertain_spans=0),
            SentenceTrace(n_uncertain_spans=1),
        ]
        mask = high_uncertainty_mask(trace)
        assert mask == [True, False, True]

    def test_rare_word_mask(self):
        freq = {"the": 100, "cat": 50, "xylophone": 1}
        mask = rare_word_mask(
            ["the cat", "a xylophone"], freq, freq_threshold=5
        )
        assert mask == [False, True]

    def test_build_word_freq(self):
        corpus = ["the cat sat", "the dog sat"]
        freq = build_word_freq(corpus)
        assert freq["the"] == 2
        assert freq["cat"] == 1


# ---------------------------------------------------------------------------
# Faithfulness audit
# ---------------------------------------------------------------------------

class TestFaithfulnessAudit:
    def _make_trace(self):
        trace = RunTrace()
        s1 = SentenceTrace(decision_mode="span_choice")
        s1.spans = [
            SpanTrace(llm_decision={
                "mode": "span_choice",
                "changed_from_top1": True,
                "change_was_correct": True,
            }),
            SpanTrace(llm_decision={
                "mode": "span_choice",
                "changed_from_top1": True,
                "change_was_correct": False,
            }),
            SpanTrace(llm_decision={
                "mode": "span_choice",
                "changed_from_top1": False,
            }),
        ]
        s2 = SentenceTrace(decision_mode="nbest_rescore")
        s2.spans = [
            SpanTrace(llm_decision={
                "mode": "nbest_rescore",
                "changed_from_top1": True,
                "change_was_correct": True,
            }),
        ]
        trace.sentences = [s1, s2]
        return trace

    def test_total_changes(self):
        audit = faithfulness_audit(self._make_trace())
        assert audit["total"]["changes"] == 3

    def test_per_mode_counts(self):
        audit = faithfulness_audit(self._make_trace())
        assert audit["span_choice"]["changes"] == 2
        assert audit["nbest_rescore"]["changes"] == 1

    def test_accuracy(self):
        audit = faithfulness_audit(self._make_trace())
        assert abs(audit["span_choice"]["accuracy"] - 0.5) < 1e-6
        assert audit["nbest_rescore"]["accuracy"] == 1.0

    def test_empty_trace(self):
        audit = faithfulness_audit(RunTrace())
        assert audit["total"]["changes"] == 0
        assert audit["total"]["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Mode breakdown WER
# ---------------------------------------------------------------------------

class TestModeBreakdownWER:
    def test_basic(self):
        trace = RunTrace()
        trace.sentences = [
            SentenceTrace(
                decision_mode="kept_top1",
                final_decoded="the cat sat",
                ground_truth="the cat sat",
            ),
            SentenceTrace(
                decision_mode="span_choice",
                final_decoded="the dog sat",
                ground_truth="the cat sat",
            ),
        ]
        breakdown = mode_breakdown_wer(trace)
        assert breakdown["kept_top1"] == 0.0
        assert breakdown["span_choice"] > 0

    def test_no_ground_truth(self):
        trace = RunTrace()
        trace.sentences = [
            SentenceTrace(
                decision_mode="kept_top1",
                final_decoded="hello",
                ground_truth="",
            ),
        ]
        breakdown = mode_breakdown_wer(trace)
        assert breakdown["kept_top1"] is None
