#!/usr/bin/env python3
"""
Load baseline and confusion-RAG result JSONs and produce a comparison table.

Usage:
    python compare_results.py --results_dir ./results
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare decoding results")
    parser.add_argument("--results_dir", default="./results")
    args = parser.parse_args()

    baselines_path = os.path.join(args.results_dir, "baselines.json")
    rag_path = os.path.join(args.results_dir, "confusion_rag.json")

    rows = []

    if os.path.exists(baselines_path):
        for entry in _load_json(baselines_path):
            ci = entry.get("wer_ci_95", [None, None])
            rows.append({
                "Method": entry["name"],
                "WER": entry["wer"],
                "WER CI low": ci[0],
                "WER CI high": ci[1],
                "CER": entry.get("cer"),
                "Oracle WER": entry.get("oracle_wer"),
                "Gap Closed": entry.get("oracle_gap_closed"),
                "Time (s)": entry.get("time_s"),
            })

    if os.path.exists(rag_path):
        entry = _load_json(rag_path)
        ci = entry.get("wer_ci_95", [None, None])
        rows.append({
            "Method": "confusion_rag",
            "WER": entry["wer"],
            "WER CI low": ci[0],
            "WER CI high": ci[1],
            "CER": entry.get("cer"),
            "Oracle WER": entry.get("oracle_wer"),
            "Gap Closed": entry.get("oracle_gap_closed"),
            "Time (s)": entry.get("time_s"),
        })

    if not rows:
        print("No result files found in", args.results_dir)
        return

    try:
        from tabulate import tabulate
        print(tabulate(rows, headers="keys", floatfmt=".4f", tablefmt="github"))
    except ImportError:
        # Fallback: simple print
        header = list(rows[0].keys())
        print(" | ".join(f"{h:>14s}" for h in header))
        print("-" * (16 * len(header)))
        for row in rows:
            vals = []
            for h in header:
                v = row[h]
                if isinstance(v, float):
                    vals.append(f"{v:>14.4f}")
                elif v is None:
                    vals.append(f"{'N/A':>14s}")
                else:
                    vals.append(f"{str(v):>14s}")
            print(" | ".join(vals))

    # Print detailed confusion-RAG info if available
    if os.path.exists(rag_path):
        entry = _load_json(rag_path)
        print("\n=== Confusion-RAG Detail ===")
        print(f"  High-uncertainty WER: {entry.get('high_uncertainty_wer')}")
        print(f"  Rare-word WER:        {entry.get('rare_word_wer')}")
        print(f"  Mode breakdown WER:   {entry.get('mode_breakdown_wer')}")
        faith = entry.get("faithfulness", {}).get("total", {})
        if faith:
            print(f"  LLM changes:          {faith.get('changes', 0)}")
            print(f"  Change accuracy:      {faith.get('accuracy', 0):.3f}")
        summary = entry.get("summary", {})
        if summary:
            print(f"  Sentences w/ unc.:    {summary.get('sentences_with_uncertain_spans', 0)}")
            print(f"  Span-choice used:     {summary.get('sentences_using_span_choice', 0)}")
            print(f"  N-best rescore used:  {summary.get('sentences_using_nbest_rescore', 0)}")
            print(f"  Kept top-1:           {summary.get('sentences_kept_top1', 0)}")


if __name__ == "__main__":
    main()
