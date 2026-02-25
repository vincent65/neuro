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
    parser.add_argument(
        "--allow_no_retrieval",
        action="store_true",
        help="Allow running without a retrieval corpus (ablation/debug only).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    nbest_outputs = np.load(args.nbest_path, allow_pickle=True)
    inference_out = np.load(args.inf_path, allow_pickle=True).item()
    reference = _extract_transcriptions(inference_out)

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
        trace_enabled=True,
        trace_dir=args.trace_dir,
    )

    # Build retriever
    retriever = Retriever(
        corpus, top_k=cfg.retrieval_top_k,
        context_window=cfg.retrieval_context_window,
        session_memory_size=cfg.session_memory_size,
    ) if corpus else None

    # Build LLM
    print(f"Loading LLM: {args.llm_name}")
    if args.llm_name.startswith("facebook/opt"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.llm_name)
        tok.padding_side = "right"
        tok.pad_token = tok.eos_token
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_name, device_map="auto", torch_dtype="auto"
        )
    else:
        from transformers import GPT2TokenizerFast, AutoModelForCausalLM
        tok = GPT2TokenizerFast.from_pretrained(args.llm_name)
        tok.padding_side = "right"
        tok.pad_token = tok.eos_token
        llm = AutoModelForCausalLM.from_pretrained(args.llm_name)

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
