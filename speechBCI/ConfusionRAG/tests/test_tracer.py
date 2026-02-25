"""Tests for tracer.py: JSON trace serialization, summary computation, mode tags."""

import json
import os
import tempfile

import pytest

from confusionrag.tracer import RunSummary, RunTrace, SentenceTrace, SpanTrace


# ---------------------------------------------------------------------------
# SpanTrace
# ---------------------------------------------------------------------------

class TestSpanTrace:
    def test_to_dict(self):
        sp = SpanTrace(
            span_start=1, span_end=2, top1_word="green",
            confusion_candidates=[{"word": "green", "weight": 0.6}],
            uncertainty_metrics={"entropy": 0.8},
            gate_result="uncertain",
            llm_decision={"mode": "span_choice", "selected": "greene"},
        )
        d = sp.to_dict()
        assert d["span_start"] == 1
        assert d["gate_result"] == "uncertain"
        assert d["llm_decision"]["mode"] == "span_choice"

    def test_defaults(self):
        sp = SpanTrace()
        d = sp.to_dict()
        assert d["span_start"] == 0
        assert d["retrieval"] is None
        assert d["llm_decision"] is None


# ---------------------------------------------------------------------------
# SentenceTrace
# ---------------------------------------------------------------------------

class TestSentenceTrace:
    def test_timer(self):
        st = SentenceTrace()
        st.start_timer()
        import time
        time.sleep(0.01)
        st.stop_timer()
        assert st.total_time_ms > 0

    def test_to_dict_excludes_internal(self):
        st = SentenceTrace(sentence_idx=0, decision_mode="span_choice")
        d = st.to_dict()
        assert "_start_time" not in d
        assert d["decision_mode"] == "span_choice"

    def test_spans_in_dict(self):
        sp = SpanTrace(span_start=0)
        st = SentenceTrace(spans=[sp])
        d = st.to_dict()
        assert len(d["spans"]) == 1


# ---------------------------------------------------------------------------
# RunSummary computation
# ---------------------------------------------------------------------------

class TestRunSummary:
    def _make_trace(self):
        trace = RunTrace()
        s1 = SentenceTrace(
            decision_mode="span_choice",
            n_uncertain_spans=2,
            spans=[
                SpanTrace(
                    gate_result="uncertain",
                    retrieval={"query": "q", "retrieved_docs": ["d"]},
                    llm_decision={
                        "mode": "span_choice",
                        "changed_from_top1": True,
                        "change_was_correct": True,
                    },
                ),
                SpanTrace(
                    gate_result="confident",
                ),
            ],
        )
        s2 = SentenceTrace(
            decision_mode="kept_top1",
            n_uncertain_spans=0,
            spans=[SpanTrace(gate_result="confident")],
        )
        s3 = SentenceTrace(
            decision_mode="nbest_rescore",
            n_uncertain_spans=1,
            spans=[
                SpanTrace(
                    gate_result="uncertain",
                    retrieval={"query": "q2"},
                    llm_decision={
                        "mode": "nbest_rescore",
                        "changed_from_top1": True,
                        "change_was_correct": False,
                    },
                ),
            ],
        )
        trace.sentences = [s1, s2, s3]
        return trace

    def test_total_sentences(self):
        trace = self._make_trace()
        s = trace.compute_summary()
        assert s.total_sentences == 3

    def test_mode_counts(self):
        s = self._make_trace().compute_summary()
        assert s.sentences_using_span_choice == 1
        assert s.sentences_using_nbest_rescore == 1
        assert s.sentences_kept_top1 == 1

    def test_uncertain_counts(self):
        s = self._make_trace().compute_summary()
        assert s.sentences_with_uncertain_spans == 2
        assert s.total_spans_gated_uncertain == 2

    def test_retrieval_count(self):
        s = self._make_trace().compute_summary()
        assert s.total_retrievals == 2

    def test_change_counts(self):
        s = self._make_trace().compute_summary()
        assert s.total_llm_changes == 2
        assert abs(s.llm_change_accuracy - 0.5) < 1e-6

    def test_to_dict(self):
        s = RunSummary(total_sentences=5)
        d = s.to_dict()
        assert d["total_sentences"] == 5


# ---------------------------------------------------------------------------
# RunTrace serialization
# ---------------------------------------------------------------------------

class TestRunTraceSerialization:
    def test_roundtrip(self):
        trace = RunTrace(config={"gate_threshold": 0.5})
        sp = SpanTrace(
            span_start=1, span_end=2, top1_word="green",
            gate_result="uncertain",
            llm_decision={"mode": "span_choice", "selected": "greene"},
        )
        st = SentenceTrace(
            sentence_idx=0,
            ground_truth="the green library",
            top1_hypothesis="the green library",
            final_decoded="the greene library",
            was_changed=True,
            decision_mode="span_choice",
            n_uncertain_spans=1,
            spans=[sp],
            total_time_ms=12.5,
        )
        trace.sentences = [st]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = trace.save(tmpdir)
            assert os.path.exists(path)

            loaded = RunTrace.load(path)
            assert loaded.run_id == trace.run_id
            assert len(loaded.sentences) == 1
            assert loaded.sentences[0].decision_mode == "span_choice"
            assert loaded.sentences[0].spans[0].top1_word == "green"

    def test_summary_in_saved_file(self):
        trace = RunTrace()
        trace.sentences = [
            SentenceTrace(decision_mode="kept_top1"),
            SentenceTrace(decision_mode="span_choice", n_uncertain_spans=1),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = trace.save(tmpdir)
            with open(path) as f:
                data = json.load(f)
            assert data["summary"]["total_sentences"] == 2
            assert data["summary"]["sentences_kept_top1"] == 1

    def test_mode_tags_always_present(self):
        trace = RunTrace()
        for mode in ["kept_top1", "span_choice", "nbest_rescore"]:
            trace.sentences.append(
                SentenceTrace(decision_mode=mode)
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = trace.save(tmpdir)
            loaded = RunTrace.load(path)
            modes = {s.decision_mode for s in loaded.sentences}
            assert modes == {"kept_top1", "span_choice", "nbest_rescore"}

    def test_empty_trace_saves(self):
        trace = RunTrace()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = trace.save(tmpdir)
            loaded = RunTrace.load(path)
            assert loaded.summary.total_sentences == 0

    def test_trace_with_error_graceful(self):
        """Trace should save even with minimal/incomplete data."""
        trace = RunTrace()
        st = SentenceTrace(sentence_idx=0)  # no spans, no mode
        trace.sentences = [st]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = trace.save(tmpdir)
            assert os.path.exists(path)
            loaded = RunTrace.load(path)
            assert len(loaded.sentences) == 1
