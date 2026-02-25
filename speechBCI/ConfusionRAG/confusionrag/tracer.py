"""
Structured JSON trace file emission for full pipeline observability.

Every pipeline run produces one RunTrace containing per-sentence SentenceTrace
objects, each with per-span SpanTrace records.  All LLM mode decisions, gating
results, retrieval queries, and scoring details are recorded so the active mode
and every decision are visible in the trace output.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Span-level trace
# ---------------------------------------------------------------------------

@dataclass
class SpanTrace:
    span_start: int = 0
    span_end: int = 0
    top1_word: str = ""
    confusion_candidates: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_metrics: Dict[str, float] = field(default_factory=dict)
    gate_result: str = ""          # "uncertain" | "confident"
    gate_threshold_used: float = 0.0
    retrieval: Optional[Dict[str, Any]] = None
    llm_decision: Optional[Dict[str, Any]] = None
    time_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Sentence-level trace
# ---------------------------------------------------------------------------

@dataclass
class SentenceTrace:
    sentence_idx: int = 0
    ground_truth: str = ""
    top1_hypothesis: str = ""
    final_decoded: str = ""
    was_changed: bool = False
    decision_mode: str = ""        # "kept_top1" | "span_choice" | "nbest_rescore"
    n_uncertain_spans: int = 0
    spans: List[SpanTrace] = field(default_factory=list)
    total_time_ms: float = 0.0

    # ---- bookkeeping helpers used during pipeline execution ----
    _start_time: float = field(default=0.0, repr=False)

    def start_timer(self) -> None:
        self._start_time = time.perf_counter()

    def stop_timer(self) -> None:
        self.total_time_ms = (time.perf_counter() - self._start_time) * 1000

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_start_time", None)
        return d


# ---------------------------------------------------------------------------
# Run-level summary
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    total_sentences: int = 0
    sentences_with_uncertain_spans: int = 0
    sentences_using_span_choice: int = 0
    sentences_using_nbest_rescore: int = 0
    sentences_kept_top1: int = 0
    total_spans_evaluated: int = 0
    total_spans_gated_uncertain: int = 0
    total_retrievals: int = 0
    total_llm_changes: int = 0
    llm_change_accuracy: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Run-level trace
# ---------------------------------------------------------------------------

@dataclass
class RunTrace:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    config: Dict[str, Any] = field(default_factory=dict)
    summary: RunSummary = field(default_factory=RunSummary)
    sentences: List[SentenceTrace] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Summary computation
    # ------------------------------------------------------------------
    def compute_summary(self) -> RunSummary:
        s = RunSummary()
        s.total_sentences = len(self.sentences)

        changes_correct = 0
        changes_total = 0

        for st in self.sentences:
            if st.n_uncertain_spans > 0:
                s.sentences_with_uncertain_spans += 1

            if st.decision_mode == "span_choice":
                s.sentences_using_span_choice += 1
            elif st.decision_mode == "nbest_rescore":
                s.sentences_using_nbest_rescore += 1
            elif st.decision_mode == "kept_top1":
                s.sentences_kept_top1 += 1

            for sp in st.spans:
                s.total_spans_evaluated += 1

                if sp.gate_result == "uncertain":
                    s.total_spans_gated_uncertain += 1

                if sp.retrieval is not None:
                    s.total_retrievals += 1

                if sp.llm_decision is not None:
                    if sp.llm_decision.get("changed_from_top1", False):
                        s.total_llm_changes += 1
                        changes_total += 1
                        if sp.llm_decision.get("change_was_correct"):
                            changes_correct += 1

        s.llm_change_accuracy = (
            changes_correct / changes_total if changes_total > 0 else 0.0
        )
        self.summary = s
        return s

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config": self.config,
            "summary": self.summary.to_dict(),
            "sentences": [st.to_dict() for st in self.sentences],
        }

    def save(self, trace_dir: str) -> str:
        """Write trace JSON to *trace_dir* and return the file path."""
        os.makedirs(trace_dir, exist_ok=True)
        self.compute_summary()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fname = f"{ts}_{self.run_id[:8]}_run.json"
        path = os.path.join(trace_dir, fname)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "RunTrace":
        """Reconstruct a RunTrace from a saved JSON file."""
        with open(path) as f:
            data = json.load(f)

        sentences = []
        for sd in data.get("sentences", []):
            spans = [SpanTrace(**sp) for sp in sd.pop("spans", [])]
            sd.pop("_start_time", None)
            sentences.append(SentenceTrace(**sd, spans=spans))

        summary_data = data.get("summary", {})
        summary = RunSummary(**summary_data)

        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            config=data.get("config", {}),
            summary=summary,
            sentences=sentences,
        )
