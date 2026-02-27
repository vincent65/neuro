#!/usr/bin/env python3
"""
Run paired same-chunk POC experiments:
- Control: no retrieval
- Treatment: retrieval enabled

Each pair keeps the same model + interpolation knobs and only changes retrieval.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.hf_utils import add_hf_model_args, load_hf_causal_lm
from confusionrag.config import ConfusionRAGConfig
from confusionrag.eval import full_evaluation
from confusionrag.pipeline import _extract_transcriptions, decode_with_confusion_rag
from confusionrag.retriever import Retriever


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
            f"(min required: {min_avg_nbest})."
        )


def _parse_csv(raw: str, cast):
    out = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(cast(x))
    if not out:
        raise ValueError("CSV argument parsed to an empty list.")
    return out


def _save_json(path: str, payload) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _model_slug(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


def _run_single(
    nbest_outputs: List[List],
    inference_out: dict,
    reference: List[str],
    corpus: List[str],
    llm,
    tok,
    cfg: ConfusionRAGConfig,
    run_name: str,
    output_dir: str,
):
    retriever = None
    if cfg.retrieval_top_k > 0 and corpus:
        retriever = Retriever(
            corpus,
            top_k=cfg.retrieval_top_k,
            context_window=cfg.retrieval_context_window,
            session_memory_size=cfg.session_memory_size,
            max_query_candidates=cfg.retrieval_max_query_candidates,
            min_candidate_weight=cfg.retrieval_min_candidate_weight,
            semantic_rerank_enabled=cfg.retrieval_semantic_rerank_enabled,
            semantic_rerank_model=cfg.retrieval_semantic_rerank_model,
            semantic_rerank_top_n=cfg.retrieval_semantic_rerank_top_n,
            confusion_memory_enabled=cfg.confusion_memory_enabled,
            confusion_memory_top_k=cfg.confusion_memory_top_k,
            confusion_memory_window=cfg.confusion_memory_window,
            phonetic_retrieval_enabled=cfg.phonetic_retrieval_enabled,
            phonetic_retrieval_top_k=cfg.phonetic_retrieval_top_k,
        )

    t0 = time.time()
    result = decode_with_confusion_rag(
        list(nbest_outputs),
        inference_out,
        llm,
        tok,
        retriever,
        cfg,
    )
    elapsed = time.time() - t0

    report = full_evaluation(
        decoded=result["decoded_transcripts"],
        reference=reference,
        nbest_outputs=list(nbest_outputs),
        trace=result["trace"],
        corpus=corpus or None,
    )
    report["time_s"] = elapsed
    report["config"] = cfg.to_dict()
    report["trace_path"] = result.get("trace_path")

    out_path = os.path.join(output_dir, f"{run_name}.json")
    _save_json(out_path, report)
    return report, out_path


def main():
    parser = argparse.ArgumentParser(description="Run paired RAG POC experiments")
    parser.add_argument("--nbest_path", required=True)
    parser.add_argument("--inf_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--output_dir", default="./results_poc_pair")
    parser.add_argument("--trace_root", default="./traces_poc_pair")
    parser.add_argument("--llm_names", default="gpt2")
    parser.add_argument("--llm_mode", default="nbest_rescore", choices=["nbest_rescore", "span_choice"])
    parser.add_argument("--gate_metric", default="entropy", choices=["entropy", "margin", "disagreement_mass"])
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument("--llm_alpha_values", default="0.5")
    parser.add_argument("--acoustic_scale_values", default="0.5")
    parser.add_argument("--retrieval_top_k_values", default="1,2")
    parser.add_argument("--evidence_max_docs_values", default="2,3")
    parser.add_argument("--retrieval_quality_min_score_gap_values", default="0.4,0.6")
    parser.add_argument("--retrieval_quality_min_top_score", type=float, default=0.0)
    parser.add_argument("--retrieval_quality_min_nonzero_docs", type=int, default=1)
    parser.add_argument("--retrieval_max_query_candidates", type=int, default=6)
    parser.add_argument("--retrieval_min_candidate_weight", type=float, default=0.02)
    parser.add_argument("--session_memory_size", type=int, default=0)
    parser.add_argument("--nbest_change_margin_threshold", type=float, default=0.0)
    parser.add_argument("--min_avg_nbest", type=int, default=5)
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="Optional cap on number of parameter pairs (<=0 means no cap).",
    )
    add_hf_model_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.trace_root, exist_ok=True)

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

    llm_names = _parse_csv(args.llm_names, str)
    llm_alphas = _parse_csv(args.llm_alpha_values, float)
    acoustic_scales = _parse_csv(args.acoustic_scale_values, float)
    retrieval_top_ks = _parse_csv(args.retrieval_top_k_values, int)
    evidence_caps = _parse_csv(args.evidence_max_docs_values, int)
    score_gaps = _parse_csv(args.retrieval_quality_min_score_gap_values, float)

    summary_rows = []
    pair_counter = 0

    for llm_name in llm_names:
        print(f"\nLoading model: {llm_name}")
        llm, tok = load_hf_causal_lm(llm_name, args)

        combos = itertools.product(
            llm_alphas,
            acoustic_scales,
            retrieval_top_ks,
            evidence_caps,
            score_gaps,
        )
        for alpha, acoustic_scale, retrieval_top_k, evidence_max_docs, score_gap in combos:
            if args.max_pairs > 0 and pair_counter >= args.max_pairs:
                break

            pair_id = f"pair_{pair_counter:03d}_{_model_slug(llm_name)}"
            control_trace_dir = os.path.join(args.trace_root, f"{pair_id}_control")
            treatment_trace_dir = os.path.join(args.trace_root, f"{pair_id}_treatment")
            os.makedirs(control_trace_dir, exist_ok=True)
            os.makedirs(treatment_trace_dir, exist_ok=True)

            base_cfg = dict(
                llm_mode=args.llm_mode,
                gate_metric=args.gate_metric,
                gate_threshold=args.gate_threshold,
                llm_alpha=alpha,
                acoustic_scale=acoustic_scale,
                nbest_change_margin_threshold=args.nbest_change_margin_threshold,
                llm_batch_size=args.llm_batch_size,
            )

            control_cfg = ConfusionRAGConfig(
                **base_cfg,
                retrieval_top_k=0,
                session_memory_size=0,
                trace_enabled=True,
                trace_dir=control_trace_dir,
            )
            treatment_cfg = ConfusionRAGConfig(
                **base_cfg,
                retrieval_top_k=retrieval_top_k,
                session_memory_size=args.session_memory_size,
                retrieval_max_query_candidates=args.retrieval_max_query_candidates,
                retrieval_min_candidate_weight=args.retrieval_min_candidate_weight,
                retrieval_quality_gate_enabled=True,
                retrieval_quality_min_top_score=args.retrieval_quality_min_top_score,
                retrieval_quality_min_score_gap=score_gap,
                retrieval_quality_min_nonzero_docs=args.retrieval_quality_min_nonzero_docs,
                evidence_max_docs=evidence_max_docs,
                trace_enabled=True,
                trace_dir=treatment_trace_dir,
            )

            print(
                f"Running {pair_id}: "
                f"alpha={alpha}, ac={acoustic_scale}, top_k={retrieval_top_k}, "
                f"evidence_max_docs={evidence_max_docs}, min_gap={score_gap}"
            )
            control_report, control_path = _run_single(
                nbest_outputs,
                inference_out,
                reference,
                corpus,
                llm,
                tok,
                control_cfg,
                run_name=f"{pair_id}_control",
                output_dir=args.output_dir,
            )
            treatment_report, treatment_path = _run_single(
                nbest_outputs,
                inference_out,
                reference,
                corpus,
                llm,
                tok,
                treatment_cfg,
                run_name=f"{pair_id}_treatment",
                output_dir=args.output_dir,
            )

            delta_wer = treatment_report["wer"] - control_report["wer"]
            summary_rows.append(
                {
                    "pair_id": pair_id,
                    "llm_name": llm_name,
                    "llm_alpha": alpha,
                    "acoustic_scale": acoustic_scale,
                    "retrieval_top_k": retrieval_top_k,
                    "evidence_max_docs": evidence_max_docs,
                    "retrieval_quality_min_score_gap": score_gap,
                    "control_wer": control_report["wer"],
                    "treatment_wer": treatment_report["wer"],
                    "delta_wer_treatment_minus_control": delta_wer,
                    "control_json": control_path,
                    "treatment_json": treatment_path,
                    "control_trace": control_report.get("trace_path"),
                    "treatment_trace": treatment_report.get("trace_path"),
                }
            )
            print(
                f"Completed {pair_id}: control={control_report['wer']:.4f}, "
                f"treatment={treatment_report['wer']:.4f}, delta={delta_wer:+.4f}"
            )
            pair_counter += 1

    if not summary_rows:
        raise RuntimeError("No experiment pairs were executed.")

    summary_rows.sort(key=lambda row: row["delta_wer_treatment_minus_control"])
    best = summary_rows[0]
    summary = {
        "success_criterion": "treatment_wer < control_wer",
        "n_pairs": len(summary_rows),
        "best_pair": best,
        "all_pairs": summary_rows,
    }

    summary_path = os.path.join(args.output_dir, "poc_pair_summary.json")
    _save_json(summary_path, summary)

    print("\n=== Best Pair ===")
    print(f"pair_id:    {best['pair_id']}")
    print(f"model:      {best['llm_name']}")
    print(f"control:    {best['control_wer']:.4f}")
    print(f"treatment:  {best['treatment_wer']:.4f}")
    print(f"delta:      {best['delta_wer_treatment_minus_control']:+.4f}")
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
