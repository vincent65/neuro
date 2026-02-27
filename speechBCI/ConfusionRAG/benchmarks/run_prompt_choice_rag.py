#!/usr/bin/env python3
"""
Run prompt-choice RAG decoding with OpenAI GPT models and compare to saved baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.openai_prompt_choice import choose_candidate_with_openai
from confusionrag.config import ConfusionRAGConfig
from confusionrag.confusion_set import build_confusion_sets
from confusionrag.constrained_llm import (
    _passes_retrieval_quality_gate,
    _retrieval_quality_stats,
)
from confusionrag.eval import full_evaluation
from confusionrag.pipeline import _extract_transcriptions
from confusionrag.retriever import RetrievalResult, Retriever
from confusionrag.tracer import RunTrace, SentenceTrace, SpanTrace


def _validate_inputs(nbest_outputs, inference_out, reference, min_avg_nbest: int) -> None:
    n_nbest = len(nbest_outputs)
    n_ref = len(reference)
    n_trans = len(inference_out.get("transcriptions", []))
    if n_nbest != n_trans or n_nbest != n_ref:
        raise ValueError(
            "Input length mismatch: "
            f"len(nbest_outputs)={n_nbest}, "
            f"len(inference_out['transcriptions'])={n_trans}, "
            f"len(reference)={n_ref}."
        )

    nbest_sizes = [len(nbest) for nbest in nbest_outputs]
    if any(size == 0 for size in nbest_sizes):
        empty = sum(1 for size in nbest_sizes if size == 0)
        raise ValueError(f"Found {empty} empty N-best entries.")

    avg_nbest = float(np.mean(nbest_sizes)) if nbest_sizes else 0.0
    if min_avg_nbest > 0 and avg_nbest < float(min_avg_nbest):
        raise ValueError(
            f"Average N-best size is too small for quality benchmarking: {avg_nbest:.2f} "
            f"(min required: {min_avg_nbest}). "
            "Regenerate inference artifacts with a larger --nbest (e.g., 100), "
            "or set --min_avg_nbest 0 to bypass this check."
        )


def _clean_hypothesis(text: str) -> str:
    h = text.strip()
    h = h.replace(">", "").replace("  ", " ").replace(" ,", ",")
    h = h.replace(" .", ".").replace(" ?", "?")
    return h


def _to_weights(span_weights: Sequence[Any]) -> List[float]:
    return [float(w) for w in span_weights]


def _mask_sentence(words: List[str], span_start: int, span_end: int) -> str:
    masked = list(words)
    masked[span_start:span_end] = ["[MASK]"]
    return " ".join(masked)


def decode_with_prompt_choice_rag(
    nbest_outputs: List[List],
    inference_out: dict,
    retriever: Optional[Retriever],
    cfg: ConfusionRAGConfig,
    openai_model: str,
    temperature: Optional[float],
    max_output_tokens: int,
    reasoning_effort: str,
    timeout_s: float,
    max_retries: int,
) -> Dict[str, Any]:
    run_trace = RunTrace(
        config={
            **cfg.to_dict(),
            "prompt_choice_model": openai_model,
            "prompt_choice_temperature": temperature,
            "prompt_choice_max_output_tokens": max_output_tokens,
            "prompt_choice_reasoning_effort": reasoning_effort,
            "prompt_choice_timeout_s": timeout_s,
            "prompt_choice_max_retries": max_retries,
        }
    )
    decoded_sentences: List[str] = []
    confidences: List[float] = []
    true_transcriptions = _extract_transcriptions(inference_out)

    for utt_idx, nbest in enumerate(nbest_outputs):
        st = SentenceTrace(sentence_idx=utt_idx)
        st.start_timer()
        if utt_idx < len(true_transcriptions):
            st.ground_truth = true_transcriptions[utt_idx]

        cs = build_confusion_sets(
            nbest,
            gate_metric=cfg.gate_metric,
            gate_threshold=cfg.gate_threshold,
            min_nbest=cfg.min_nbest,
        )
        st.top1_hypothesis = cs.top1_hypothesis
        st.n_uncertain_spans = sum(cs.is_uncertain) if cs.is_uncertain else 0

        words = cs.top1_hypothesis.split()
        llm_applied = False
        for span_idx, span in enumerate(cs.spans):
            t0 = time.perf_counter()
            top1_word = words[span.span_start] if span.span_start < len(words) else ""
            span_trace = SpanTrace(
                span_start=span.span_start,
                span_end=span.span_end,
                top1_word=top1_word,
                confusion_candidates=[
                    {"word": c, "weight": float(w)}
                    for c, w in zip(span.candidates, span.weights)
                ],
                uncertainty_metrics={
                    "entropy": span.entropy,
                    "margin": span.margin,
                    "disagreement_mass": span.disagreement_mass,
                },
                gate_result="uncertain" if cs.is_uncertain[span_idx] else "confident",
            )

            if not cs.is_uncertain[span_idx]:
                span_trace.time_ms = (time.perf_counter() - t0) * 1000
                st.spans.append(span_trace)
                continue

            rr = (
                retriever.retrieve_for_span(words, span)
                if retriever is not None
                else RetrievalResult(query="", retrieved_docs=[], scores=[], retrieval_time_ms=0.0)
            )
            span_trace.retrieval = rr.to_dict()

            docs_for_evidence = list(rr.retrieved_docs)
            quality_gate_passed = True
            if cfg.retrieval_quality_gate_enabled:
                quality_stats = _retrieval_quality_stats({0: rr})
                quality_gate_passed = _passes_retrieval_quality_gate(
                    quality_stats,
                    cfg.retrieval_quality_min_top_score,
                    cfg.retrieval_quality_min_score_gap,
                    cfg.retrieval_quality_min_nonzero_docs,
                )
                if not quality_gate_passed:
                    docs_for_evidence = []

            docs_for_evidence = docs_for_evidence[: max(cfg.evidence_max_docs, 0)]
            masked_sentence = _mask_sentence(words, span.span_start, span.span_end)
            choice = choose_candidate_with_openai(
                masked_sentence=masked_sentence,
                candidates=span.candidates,
                candidate_weights=_to_weights(span.weights),
                evidence_docs=docs_for_evidence,
                model=openai_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                timeout_s=timeout_s,
                max_retries=max_retries,
            )
            selected = choice.chosen_candidate
            words[span.span_start:span.span_end] = [selected]

            span_trace.llm_decision = {
                "mode": "prompt_choice",
                "candidate_scores": [
                    {"candidate": c, "prior_weight": float(w)}
                    for c, w in zip(span.candidates, span.weights)
                ],
                "selected": selected,
                "changed_from_top1": selected != top1_word,
                "change_was_correct": None,
                "retrieval_quality_gate_enabled": cfg.retrieval_quality_gate_enabled,
                "retrieval_quality_gate_passed": quality_gate_passed,
                "evidence_docs_used": len(docs_for_evidence),
                "prompt_confidence": choice.confidence,
                "used_fallback": choice.used_fallback,
                "fallback_reason": choice.fallback_reason,
                "raw_response_text": choice.raw_response_text,
                "error": choice.error,
            }
            llm_applied = True
            span_trace.time_ms = (time.perf_counter() - t0) * 1000
            st.spans.append(span_trace)

        decoded = " ".join(words)
        st.final_decoded = decoded
        st.was_changed = decoded != cs.top1_hypothesis
        st.decision_mode = "prompt_choice" if llm_applied else "kept_top1"

        # Align confidence to prior weighted uncertainty (prompt APIs typically
        # do not expose calibrated token probabilities).
        confidences.append(1.0 if len(cs.nbest_scores) <= 1 else float(np.max(_softmax_scores(cs.nbest_scores))))

        if st.ground_truth:
            gt_words = st.ground_truth.split()
            for sp in st.spans:
                if sp.llm_decision is None:
                    continue
                selected = sp.llm_decision.get("selected", "")
                if sp.span_start < len(gt_words):
                    sp.llm_decision["change_was_correct"] = (
                        selected.lower() == gt_words[sp.span_start].lower()
                    )

        st.stop_timer()
        run_trace.sentences.append(st)
        decoded_sentences.append(decoded)

        if retriever is not None:
            retriever.add_to_session_memory(decoded)

    trace_path = None
    if cfg.trace_enabled:
        trace_path = run_trace.save(cfg.trace_dir)
    return {
        "decoded_transcripts": decoded_sentences,
        "confidences": confidences,
        "trace_path": trace_path,
        "trace": run_trace,
    }


def _softmax_scores(nbest_scores: Sequence[Sequence[float]]) -> np.ndarray:
    combined = np.array([float(s[0]) + float(s[1]) for s in nbest_scores], dtype=float)
    combined -= np.max(combined)
    probs = np.exp(combined)
    return probs / np.sum(probs)


def _parse_csv_paths(raw: str) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_metrics(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        if "wer" in payload:
            return [{
                "source_path": path,
                "name": payload.get("name", os.path.basename(path)),
                "wer": payload.get("wer"),
                "cer": payload.get("cer"),
            }]
        return []

    if isinstance(payload, list):
        out = []
        for idx, row in enumerate(payload):
            if isinstance(row, dict) and "wer" in row:
                out.append({
                    "source_path": path,
                    "name": row.get("name", f"{os.path.basename(path)}[{idx}]"),
                    "wer": row.get("wer"),
                    "cer": row.get("cer"),
                })
        return out
    return []


def _write_comparison_report(
    report_path: str,
    run_name: str,
    run_metrics: Dict[str, Any],
    baseline_rows: List[Dict[str, Any]],
) -> None:
    lines = [
        "# Prompt-Choice Smoke Comparison",
        "",
        f"- run: `{run_name}`",
        f"- wer: `{run_metrics['wer']:.6f}`",
        f"- cer: `{run_metrics['cer']:.6f}`",
        "",
        "## Baseline Comparison",
    ]
    if not baseline_rows:
        lines.extend(["", "No baseline rows were loaded from --compare_jsons."])
    else:
        lines.append("")
        lines.append("| baseline | source | wer | delta_wer (run-baseline) | cer | delta_cer (run-baseline) |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in baseline_rows:
            b_wer = row.get("wer")
            b_cer = row.get("cer")
            d_wer = (run_metrics["wer"] - b_wer) if isinstance(b_wer, (int, float)) else None
            d_cer = (run_metrics["cer"] - b_cer) if isinstance(b_cer, (int, float)) else None
            lines.append(
                f"| {row.get('name')} | `{row.get('source_path')}` | "
                f"{_fmt_num(b_wer)} | {_fmt_num(d_wer)} | {_fmt_num(b_cer)} | {_fmt_num(d_cer)} |"
            )
    lines.append("")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def _fmt_num(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt-choice RAG benchmark")
    parser.add_argument("--nbest_path", required=True)
    parser.add_argument("--inf_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--output_dir", default="./results_prompt_choice")
    parser.add_argument("--trace_dir", default="./traces_prompt_choice")
    parser.add_argument("--run_name", default="prompt_choice_rag")
    parser.add_argument("--openai_model", default="gpt-5")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature; omit for models that do not support it.",
    )
    parser.add_argument("--max_output_tokens", type=int, default=128)
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="minimal",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort for reasoning-capable OpenAI models.",
    )
    parser.add_argument("--openai_timeout_s", type=float, default=30.0)
    parser.add_argument("--openai_max_retries", type=int, default=2)
    parser.add_argument("--gate_metric", default="entropy", choices=["entropy", "margin", "disagreement_mass"])
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument("--retrieval_top_k", type=int, default=2)
    parser.add_argument("--retrieval_context_window", type=int, default=5)
    parser.add_argument("--retrieval_max_query_candidates", type=int, default=6)
    parser.add_argument("--retrieval_min_candidate_weight", type=float, default=0.02)
    parser.add_argument("--session_memory_size", type=int, default=0)
    parser.add_argument("--evidence_max_docs", type=int, default=2)
    parser.add_argument("--retrieval_quality_gate_enabled", action="store_true")
    parser.add_argument("--retrieval_quality_min_top_score", type=float, default=0.0)
    parser.add_argument("--retrieval_quality_min_score_gap", type=float, default=0.6)
    parser.add_argument("--retrieval_quality_min_nonzero_docs", type=int, default=1)
    parser.add_argument("--min_avg_nbest", type=int, default=5)
    parser.add_argument(
        "--compare_jsons",
        default="",
        help="Comma-separated existing result JSON paths for side-by-side comparison.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Warning: OPENAI_API_KEY is not set. "
            "Prompt-choice will run in deterministic fallback mode."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.trace_dir, exist_ok=True)

    nbest_outputs = np.load(args.nbest_path, allow_pickle=True)
    inference_out = np.load(args.inf_path, allow_pickle=True).item()
    reference = _extract_transcriptions(inference_out)
    _validate_inputs(nbest_outputs, inference_out, reference, args.min_avg_nbest)

    if not os.path.exists(args.corpus_path):
        raise ValueError(f"Corpus file not found: {args.corpus_path}")
    with open(args.corpus_path) as f:
        corpus = [line.strip() for line in f if line.strip()]
    if not corpus:
        raise ValueError("Corpus file is empty.")

    cfg = ConfusionRAGConfig(
        llm_mode="span_choice",
        gate_metric=args.gate_metric,
        gate_threshold=args.gate_threshold,
        retrieval_top_k=args.retrieval_top_k,
        retrieval_context_window=args.retrieval_context_window,
        retrieval_max_query_candidates=args.retrieval_max_query_candidates,
        retrieval_min_candidate_weight=args.retrieval_min_candidate_weight,
        retrieval_quality_gate_enabled=args.retrieval_quality_gate_enabled,
        retrieval_quality_min_top_score=args.retrieval_quality_min_top_score,
        retrieval_quality_min_score_gap=args.retrieval_quality_min_score_gap,
        retrieval_quality_min_nonzero_docs=args.retrieval_quality_min_nonzero_docs,
        session_memory_size=args.session_memory_size,
        evidence_max_docs=args.evidence_max_docs,
        trace_enabled=True,
        trace_dir=args.trace_dir,
    )
    retriever = Retriever(
        corpus,
        top_k=cfg.retrieval_top_k,
        context_window=cfg.retrieval_context_window,
        session_memory_size=cfg.session_memory_size,
        max_query_candidates=cfg.retrieval_max_query_candidates,
        min_candidate_weight=cfg.retrieval_min_candidate_weight,
    )

    print(
        "Running prompt-choice RAG "
        f"(model={args.openai_model}, gate={cfg.gate_metric}@{cfg.gate_threshold}, "
        f"retrieval_top_k={cfg.retrieval_top_k}, evidence_max_docs={cfg.evidence_max_docs})"
    )
    t0 = time.time()
    result = decode_with_prompt_choice_rag(
        list(nbest_outputs),
        inference_out,
        retriever,
        cfg,
        openai_model=args.openai_model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        timeout_s=args.openai_timeout_s,
        max_retries=args.openai_max_retries,
    )
    elapsed = time.time() - t0
    print(f"Prompt-choice pipeline completed in {elapsed:.1f}s")
    if result["trace_path"]:
        print(f"Trace saved to: {result['trace_path']}")

    report = full_evaluation(
        decoded=result["decoded_transcripts"],
        reference=reference,
        nbest_outputs=list(nbest_outputs),
        trace=result["trace"],
        corpus=corpus or None,
    )
    report["name"] = args.run_name
    report["time_s"] = elapsed
    report["config"] = {
        **cfg.to_dict(),
        "openai_model": args.openai_model,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "openai_timeout_s": args.openai_timeout_s,
        "openai_max_retries": args.openai_max_retries,
    }
    report["trace_path"] = result.get("trace_path")

    out_json = os.path.join(args.output_dir, f"{args.run_name}.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Result JSON saved to: {out_json}")

    comparison_rows: List[Dict[str, Any]] = []
    for path in _parse_csv_paths(args.compare_jsons):
        if not os.path.exists(path):
            print(f"Skipping missing compare JSON: {path}")
            continue
        comparison_rows.extend(_load_metrics(path))

    report_md = os.path.join(args.output_dir, f"{args.run_name}_comparison.md")
    _write_comparison_report(
        report_path=report_md,
        run_name=args.run_name,
        run_metrics=report,
        baseline_rows=comparison_rows,
    )
    print(f"Comparison markdown saved to: {report_md}")


if __name__ == "__main__":
    main()
