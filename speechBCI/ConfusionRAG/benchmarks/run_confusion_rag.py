#!/usr/bin/env python3
"""
Run the full confusion-set guided RAG decoding pipeline and save results.

Usage:
    python run_confusion_rag.py \
        --nbest_path  /path/to/nbest_outputs.npy \
        --inf_path    /path/to/inference_out.npy \
        --corpus_path /path/to/train_transcriptions.txt \
        --output_dir  ./results \
        --trace_dir   ./traces \
        --llm_name    facebook/opt-6.7b \
        --llm_mode    nbest_rescore \
        --gate_metric entropy \
        --gate_threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from confusionrag.config import ConfusionRAGConfig
from confusionrag.eval import full_evaluation
from confusionrag.pipeline import decode_with_confusion_rag, _extract_transcriptions
from confusionrag.retriever import Retriever
from benchmarks.hf_utils import add_hf_model_args, load_hf_causal_lm


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


def main():
    parser = argparse.ArgumentParser(description="Run confusion-set guided RAG")
    parser.add_argument("--nbest_path", required=True)
    parser.add_argument("--inf_path", required=True)
    parser.add_argument("--corpus_path", default=None)
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--trace_dir", default="./traces")
    parser.add_argument("--llm_name", default="gpt2")
    parser.add_argument("--llm_mode", default="nbest_rescore",
                        choices=["nbest_rescore", "span_choice"])
    parser.add_argument("--gate_metric", default="entropy",
                        choices=["entropy", "margin", "disagreement_mass"])
    parser.add_argument("--gate_threshold", type=float, default=0.5)
    parser.add_argument("--llm_alpha", type=float, default=0.5)
    parser.add_argument("--acoustic_scale", type=float, default=0.5)
    parser.add_argument("--retrieval_top_k", type=int, default=5)
    parser.add_argument("--retrieval_max_query_candidates", type=int, default=6)
    parser.add_argument("--retrieval_min_candidate_weight", type=float, default=0.02)
    parser.add_argument("--session_memory_size", type=int, default=10)
    parser.add_argument("--evidence_max_docs", type=int, default=5)
    parser.add_argument("--retrieval_quality_gate_enabled", action="store_true")
    parser.add_argument("--retrieval_quality_min_top_score", type=float, default=0.0)
    parser.add_argument("--retrieval_quality_min_score_gap", type=float, default=0.0)
    parser.add_argument("--retrieval_quality_min_nonzero_docs", type=int, default=0)
    parser.add_argument("--retrieval_semantic_rerank_enabled", action="store_true")
    parser.add_argument(
        "--retrieval_semantic_rerank_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--retrieval_semantic_rerank_top_n", type=int, default=20)
    parser.add_argument("--confusion_memory_enabled", action="store_true")
    parser.add_argument("--confusion_memory_top_k", type=int, default=0)
    parser.add_argument("--confusion_memory_window", type=int, default=2)
    parser.add_argument("--phonetic_retrieval_enabled", action="store_true")
    parser.add_argument("--phonetic_retrieval_top_k", type=int, default=0)
    parser.add_argument("--nbest_change_margin_threshold", type=float, default=0.0)
    parser.add_argument(
        "--min_avg_nbest",
        type=int,
        default=5,
        help="Fail fast if the average N-best size is below this value (set 0 to disable).",
    )
    parser.add_argument(
        "--allow_no_retrieval",
        action="store_true",
        help="Allow running without a retrieval corpus (ablation/debug only).",
    )
    add_hf_model_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    nbest_outputs = np.load(args.nbest_path, allow_pickle=True)
    inference_out = np.load(args.inf_path, allow_pickle=True).item()
    reference = _extract_transcriptions(inference_out)
    _validate_inputs(nbest_outputs, inference_out, reference, args.min_avg_nbest)

    corpus = []
    if args.corpus_path and os.path.exists(args.corpus_path):
        with open(args.corpus_path) as f:
            corpus = [line.strip() for line in f if line.strip()]
    if not corpus and not args.allow_no_retrieval:
        raise ValueError(
            "No retrieval corpus loaded. Provide --corpus_path with training "
            "transcriptions, or pass --allow_no_retrieval for ablation/debug runs."
        )

    # Build config
    cfg = ConfusionRAGConfig(
        llm_mode=args.llm_mode,
        gate_metric=args.gate_metric,
        gate_threshold=args.gate_threshold,
        llm_alpha=args.llm_alpha,
        acoustic_scale=args.acoustic_scale,
        retrieval_top_k=args.retrieval_top_k,
        retrieval_max_query_candidates=args.retrieval_max_query_candidates,
        retrieval_min_candidate_weight=args.retrieval_min_candidate_weight,
        session_memory_size=args.session_memory_size,
        retrieval_quality_gate_enabled=args.retrieval_quality_gate_enabled,
        retrieval_quality_min_top_score=args.retrieval_quality_min_top_score,
        retrieval_quality_min_score_gap=args.retrieval_quality_min_score_gap,
        retrieval_quality_min_nonzero_docs=args.retrieval_quality_min_nonzero_docs,
        retrieval_semantic_rerank_enabled=args.retrieval_semantic_rerank_enabled,
        retrieval_semantic_rerank_model=args.retrieval_semantic_rerank_model,
        retrieval_semantic_rerank_top_n=args.retrieval_semantic_rerank_top_n,
        confusion_memory_enabled=args.confusion_memory_enabled,
        confusion_memory_top_k=args.confusion_memory_top_k,
        confusion_memory_window=args.confusion_memory_window,
        phonetic_retrieval_enabled=args.phonetic_retrieval_enabled,
        phonetic_retrieval_top_k=args.phonetic_retrieval_top_k,
        nbest_change_margin_threshold=args.nbest_change_margin_threshold,
        evidence_max_docs=args.evidence_max_docs,
        llm_batch_size=args.llm_batch_size,
        trace_enabled=True,
        trace_dir=args.trace_dir,
    )

    # Build retriever
    retriever = Retriever(
        corpus, top_k=cfg.retrieval_top_k,
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
    ) if corpus else None

    # Build LLM
    print(f"Loading LLM: {args.llm_name}")
    llm, tok = load_hf_causal_lm(args.llm_name, args)

    # Run pipeline
    print(f"Running confusion-set RAG (mode={cfg.llm_mode}, gate={cfg.gate_metric}@{cfg.gate_threshold})")
    t0 = time.time()
    result = decode_with_confusion_rag(
        list(nbest_outputs), inference_out, llm, tok, retriever, cfg,
    )
    elapsed = time.time() - t0
    print(f"Pipeline completed in {elapsed:.1f}s")

    if result["trace_path"]:
        print(f"Trace saved to: {result['trace_path']}")

    # Full evaluation
    report = full_evaluation(
        decoded=result["decoded_transcripts"],
        reference=reference,
        nbest_outputs=list(nbest_outputs),
        trace=result["trace"],
        corpus=corpus or None,
    )
    report["time_s"] = elapsed
    report["config"] = cfg.to_dict()

    out_path = os.path.join(args.output_dir, "confusion_rag.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n=== Results ===")
    print(f"  WER:                {report['wer']:.4f}  ({report['wer_ci_95'][0]:.4f} - {report['wer_ci_95'][1]:.4f})")
    print(f"  CER:                {report['cer']:.4f}")
    print(f"  Top-1 WER:          {report['top1_wer']:.4f}")
    print(f"  Oracle WER:         {report['oracle_wer']:.4f}")
    print(f"  Oracle gap closed:  {report['oracle_gap_closed']:.3f}")
    print(f"  High-unc WER:       {report['high_uncertainty_wer']}")
    print(f"  Rare-word WER:      {report['rare_word_wer']}")
    print(f"\nMode breakdown WER:   {report['mode_breakdown_wer']}")
    print(f"Faithfulness:         {report['faithfulness']}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
