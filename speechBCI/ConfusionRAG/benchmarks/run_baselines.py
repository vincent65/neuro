#!/usr/bin/env python3
"""
Run all baseline decoding methods and save results to JSON.

Baselines:
  1. Neural + trigram/5gram LM  (top-1 from N-gram decoder)
  2. LLM rescore (no retrieval)
  3. Always-on RAG              (retrieval on every sentence, no gating)
  4. Context-only               (prior sentences as LLM context, no confusion set)

Usage:
    python run_baselines.py \
        --nbest_path  /path/to/nbest_outputs.npy \
        --inf_path    /path/to/inference_out.npy \
        --corpus_path /path/to/train_transcriptions.txt \
        --output_dir  ./results \
        --llm_name    facebook/opt-6.7b
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from confusionrag.config import ConfusionRAGConfig
from confusionrag.eval import (
    compute_cer,
    compute_oracle_wer,
    compute_wer,
    compute_wer_with_ci,
    oracle_gap_closed,
)
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


def _load_data(args):
    nbest_outputs = np.load(args.nbest_path, allow_pickle=True)
    inference_out = np.load(args.inf_path, allow_pickle=True).item()
    reference = _extract_transcriptions(inference_out)
    _validate_inputs(nbest_outputs, inference_out, reference, args.min_avg_nbest)

    corpus = []
    if args.corpus_path and os.path.exists(args.corpus_path):
        with open(args.corpus_path) as f:
            corpus = [line.strip() for line in f if line.strip()]
    return nbest_outputs, inference_out, reference, corpus


def _top1_from_nbest(nbest_outputs):
    decoded = []
    for nbest in nbest_outputs:
        if nbest:
            h = nbest[0][0].strip().replace(">", "").replace("  ", " ")
            h = h.replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
            decoded.append(h)
        else:
            decoded.append("")
    return decoded


def _build_llm(args):
    return load_hf_causal_lm(args.llm_name, args)


def baseline_ngram_top1(nbest_outputs, reference):
    """Baseline 1: top-1 from N-gram decoder."""
    decoded = _top1_from_nbest(nbest_outputs)
    wer, lo, hi = compute_wer_with_ci(decoded, reference)
    cer = compute_cer(decoded, reference)
    oracle = compute_oracle_wer(nbest_outputs, reference)
    return {
        "name": "ngram_top1",
        "wer": wer, "wer_ci_95": [lo, hi],
        "cer": cer, "oracle_wer": oracle,
        "oracle_gap_closed": 0.0,
    }


def baseline_llm_rescore_no_rag(
    nbest_outputs, inference_out, reference, llm, tok, llm_batch_size: int = 0
):
    """Baseline 2: LLM N-best rescoring, no retrieval."""
    cfg = ConfusionRAGConfig(
        llm_mode="nbest_rescore",
        gate_threshold=0.0,  # always triggers
        llm_batch_size=llm_batch_size,
        trace_enabled=False,
    )
    result = decode_with_confusion_rag(
        nbest_outputs, inference_out, llm, tok, retriever=None, config=cfg,
    )
    decoded = result["decoded_transcripts"]
    wer, lo, hi = compute_wer_with_ci(decoded, reference)
    cer = compute_cer(decoded, reference)
    top1_wer = compute_wer(_top1_from_nbest(nbest_outputs), reference)
    oracle = compute_oracle_wer(nbest_outputs, reference)
    return {
        "name": "llm_rescore_no_rag",
        "wer": wer, "wer_ci_95": [lo, hi],
        "cer": cer, "oracle_wer": oracle,
        "oracle_gap_closed": oracle_gap_closed(top1_wer, wer, oracle),
    }


def baseline_always_on_rag(
    nbest_outputs, inference_out, reference, llm, tok, corpus, llm_batch_size: int = 0
):
    """Baseline 3: retrieval on every sentence, no gating."""
    if not corpus:
        raise ValueError(
            "always_on_rag requires a non-empty corpus_path (training transcriptions)."
        )
    cfg = ConfusionRAGConfig(
        llm_mode="nbest_rescore",
        gate_threshold=0.0,
        llm_batch_size=llm_batch_size,
        trace_enabled=False,
    )
    retriever = Retriever(corpus, top_k=cfg.retrieval_top_k)
    result = decode_with_confusion_rag(
        nbest_outputs, inference_out, llm, tok, retriever, cfg,
    )
    decoded = result["decoded_transcripts"]
    wer, lo, hi = compute_wer_with_ci(decoded, reference)
    cer = compute_cer(decoded, reference)
    top1_wer = compute_wer(_top1_from_nbest(nbest_outputs), reference)
    oracle = compute_oracle_wer(nbest_outputs, reference)
    return {
        "name": "always_on_rag",
        "wer": wer, "wer_ci_95": [lo, hi],
        "cer": cer, "oracle_wer": oracle,
        "oracle_gap_closed": oracle_gap_closed(top1_wer, wer, oracle),
    }


def baseline_context_only(
    nbest_outputs, inference_out, reference, llm, tok, llm_batch_size: int = 0
):
    """
    Baseline 4: context-only LLM N-best rescoring.

    Uses prior decoded sentences as LLM evidence, with no retrieval corpus and
    no confusion-set gating.
    """
    from confusionrag.constrained_llm import _build_evidence_prefix, _rescore_with_llm

    cfg = ConfusionRAGConfig(trace_enabled=False, llm_batch_size=llm_batch_size)
    session_memory = deque(maxlen=cfg.session_memory_size)
    decoded = []

    for nbest in nbest_outputs:
        if not nbest:
            decoded.append("")
            continue

        hypotheses = []
        acoustic_scores = []
        old_lm_scores = []
        for sent, ac, lm in nbest:
            h = sent.strip().replace(">", "").replace("  ", " ")
            h = h.replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
            hypotheses.append(h)
            acoustic_scores.append(float(ac))
            old_lm_scores.append(float(lm))

        evidence_prefix = _build_evidence_prefix(list(session_memory))
        texts_for_llm = [evidence_prefix + h for h in hypotheses]
        llm_scores = _rescore_with_llm(
            llm,
            tok,
            texts_for_llm,
            length_penalty=cfg.llm_length_penalty,
            max_batch_size=cfg.llm_batch_size,
        )
        combined = (
            cfg.llm_alpha * np.array(llm_scores)
            + (1 - cfg.llm_alpha) * np.array(old_lm_scores)
            + cfg.acoustic_scale * np.array(acoustic_scores)
        )
        best_idx = int(np.argmax(combined))
        best = hypotheses[best_idx]
        decoded.append(best)
        session_memory.append(best)

    wer, lo, hi = compute_wer_with_ci(decoded, reference)
    cer = compute_cer(decoded, reference)
    top1_wer = compute_wer(_top1_from_nbest(nbest_outputs), reference)
    oracle = compute_oracle_wer(nbest_outputs, reference)
    return {
        "name": "context_only",
        "wer": wer, "wer_ci_95": [lo, hi],
        "cer": cer, "oracle_wer": oracle,
        "oracle_gap_closed": oracle_gap_closed(top1_wer, wer, oracle),
    }


def main():
    parser = argparse.ArgumentParser(description="Run baseline decoders")
    parser.add_argument("--nbest_path", required=True)
    parser.add_argument("--inf_path", required=True)
    parser.add_argument("--corpus_path", default=None)
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--llm_name", default="gpt2")
    parser.add_argument(
        "--min_avg_nbest",
        type=int,
        default=5,
        help="Fail fast if the average N-best size is below this value (set 0 to disable).",
    )
    add_hf_model_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    nbest_outputs, inference_out, reference, corpus = _load_data(args)

    results = []

    print("=== Baseline 1: N-gram top-1 ===")
    t0 = time.time()
    r = baseline_ngram_top1(nbest_outputs, reference)
    r["time_s"] = time.time() - t0
    results.append(r)
    print(f"  WER={r['wer']:.4f}  CER={r['cer']:.4f}")

    print("\nLoading LLM...")
    llm, tok = _build_llm(args)

    print("=== Baseline 2: LLM rescore (no RAG) ===")
    t0 = time.time()
    r = baseline_llm_rescore_no_rag(
        nbest_outputs,
        inference_out,
        reference,
        llm,
        tok,
        llm_batch_size=args.llm_batch_size,
    )
    r["time_s"] = time.time() - t0
    results.append(r)
    print(f"  WER={r['wer']:.4f}  CER={r['cer']:.4f}  gap={r['oracle_gap_closed']:.3f}")

    print("=== Baseline 3: Always-on RAG ===")
    t0 = time.time()
    r = baseline_always_on_rag(
        nbest_outputs,
        inference_out,
        reference,
        llm,
        tok,
        corpus,
        llm_batch_size=args.llm_batch_size,
    )
    r["time_s"] = time.time() - t0
    results.append(r)
    print(f"  WER={r['wer']:.4f}  CER={r['cer']:.4f}  gap={r['oracle_gap_closed']:.3f}")

    print("=== Baseline 4: Context-only ===")
    t0 = time.time()
    r = baseline_context_only(
        nbest_outputs,
        inference_out,
        reference,
        llm,
        tok,
        llm_batch_size=args.llm_batch_size,
    )
    r["time_s"] = time.time() - t0
    results.append(r)
    print(f"  WER={r['wer']:.4f}  CER={r['cer']:.4f}  gap={r['oracle_gap_closed']:.3f}")

    out_path = os.path.join(args.output_dir, "baselines.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
