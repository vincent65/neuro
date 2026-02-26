#!/usr/bin/env python3
"""
Generate publication-ready figures and metrics tables for the ConfusionRAG research paper.

Loads smoke test results and traces, then produces:
- 9 figures (method comparison, oracle gap, mode distribution, WER by mode, faithfulness,
  uncertainty gating, margin vs correctness, runtime, retrieval payload)
- metrics_table.tex
- qualitative_examples.md

Usage:
    python scripts/generate_paper_figures.py
    python scripts/generate_paper_figures.py --results_dir results_smoke --output_dir paper_figures --format pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_root = _script_dir.parent
sys.path.insert(0, str(_root))

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def _load_trace(path: str):
    """Load trace JSON (standalone, no confusionrag dependency)."""
    with open(path) as f:
        return json.load(f)


def _attr(obj, key, default=None):
    """Get attribute or dict key from obj (works for both dataclass and dict)."""
    if obj is None:
        return default
    if hasattr(obj, key):
        return getattr(obj, key, default)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_json(path: str) -> dict | list:
    with open(path) as f:
        return json.load(f)


def load_data(results_dir: str, traces_dir: str):
    """Load baselines, confusion_rag, and trace. Returns (baselines, rag, trace) or None for missing."""
    baselines_path = os.path.join(results_dir, "baselines.json")
    rag_path = os.path.join(results_dir, "confusion_rag.json")

    baselines = []
    if os.path.exists(baselines_path):
        baselines = _load_json(baselines_path)

    rag = None
    if os.path.exists(rag_path):
        rag = _load_json(rag_path)

    trace = None
    if os.path.isdir(traces_dir):
        trace_files = sorted(
            [f for f in os.listdir(traces_dir) if f.endswith("_run.json")],
            reverse=True,
        )
        if trace_files:
            trace_path = os.path.join(traces_dir, trace_files[0])
            trace = _load_trace(trace_path)

    return baselines, rag, trace


# ---------------------------------------------------------------------------
# Figure 1: Method comparison (WER, CER)
# ---------------------------------------------------------------------------


def plot_method_comparison(baselines: list, rag: dict, output_dir: str, fmt: str):
    methods = []
    wers = []
    cers = []
    ci_low = []
    ci_high = []

    for entry in baselines:
        methods.append(entry["name"])
        wers.append(entry["wer"])
        cers.append(entry.get("cer"))
        ci = entry.get("wer_ci_95", [None, None])
        ci_low.append(ci[0])
        ci_high.append(ci[1])

    if rag:
        methods.append("confusion_rag")
        wers.append(rag["wer"])
        cers.append(rag.get("cer"))
        ci = rag.get("wer_ci_95", [None, None])
        ci_low.append(ci[0])
        ci_high.append(ci[1])

    x = range(len(methods))
    width = 0.35
    cer_vals = [c if c is not None else 0 for c in cers]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], wers, width, label="WER", color="steelblue")
    ax.bar([i + width / 2 for i in x], cer_vals, width, label="CER", color="coral", alpha=0.8)

    err_lo = [w - lo if lo is not None else 0 for w, lo in zip(wers, ci_low)]
    err_hi = [hi - w if hi is not None else 0 for w, hi in zip(wers, ci_high)]
    ax.errorbar(
        [i - width / 2 for i in x],
        wers,
        yerr=[err_lo, err_hi],
        fmt="none",
        color="black",
        capsize=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Error rate")
    ax.set_title("Method Comparison: WER and CER")
    ax.legend()
    ax.set_ylim(0, max(wers) * 1.2 if wers else 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"method_comparison_wer.{fmt}"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Oracle gap closed
# ---------------------------------------------------------------------------


def plot_oracle_gap_closed(baselines: list, rag: dict, output_dir: str, fmt: str):
    methods = []
    gaps = []

    for entry in baselines:
        methods.append(entry["name"])
        gaps.append(entry.get("oracle_gap_closed", 0))

    if rag:
        methods.append("confusion_rag")
        gaps.append(rag.get("oracle_gap_closed", 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("viridis", len(methods))
    x = range(len(methods))
    ax.bar(x, gaps, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Oracle gap closed")
    ax.set_title("Fraction of Oracle Improvement Recovered")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"oracle_gap_closed.{fmt}"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Decision mode distribution
# ---------------------------------------------------------------------------


def plot_decision_mode_distribution(trace, output_dir: str, fmt: str):
    s = _attr(trace, "summary")
    labels = ["kept_top1", "nbest_rescore", "span_choice"]
    counts = [
        _attr(s, "sentences_kept_top1", 0) or 0,
        _attr(s, "sentences_using_nbest_rescore", 0) or 0,
        _attr(s, "sentences_using_span_choice", 0) or 0,
    ]
    # Filter out zero modes for cleaner pie
    data = [(l, c) for l, c in zip(labels, counts) if c > 0]
    if not data:
        return
    labels, counts = zip(*data)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Decision Mode Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"decision_mode_distribution.{fmt}"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4: WER by decision mode
# ---------------------------------------------------------------------------


def plot_wer_by_mode(rag: dict, output_dir: str, fmt: str):
    mode_wer = rag.get("mode_breakdown_wer", {})
    if not mode_wer:
        return

    modes = [k for k, v in mode_wer.items() if v is not None]
    wers = [mode_wer[k] for k in modes]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(modes, wers, color=["steelblue", "coral", "seagreen"][: len(modes)])
    ax.set_ylabel("WER")
    ax.set_title("WER by Decision Mode")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"wer_by_mode.{fmt}"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5: Faithfulness / LLM change accuracy
# ---------------------------------------------------------------------------


def plot_faithfulness_accuracy(rag: dict, output_dir: str, fmt: str):
    faith = rag.get("faithfulness", {}).get("total", {})
    correct = faith.get("correct", 0)
    incorrect = faith.get("incorrect", 0)
    if correct == 0 and incorrect == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(["Correct", "Incorrect"], [correct, incorrect], color=["seagreen", "coral"])
    ax.set_ylabel("Count")
    ax.set_title(f"LLM Changes (accuracy: {faith.get('accuracy', 0):.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"faithfulness_accuracy.{fmt}"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 6: Uncertainty metric distribution (gating)
# ---------------------------------------------------------------------------


def plot_uncertainty_gating(trace, output_dir: str, fmt: str):
    confident_entropies = []
    uncertain_entropies = []

    for st in _attr(trace, "sentences") or []:
        for sp in _attr(st, "spans") or []:
            metrics = _attr(sp, "uncertainty_metrics") or {}
            ent = metrics.get("entropy") if isinstance(metrics, dict) else _attr(metrics, "entropy")
            if ent is None:
                continue
            gate = _attr(sp, "gate_result") or ""
            if gate == "confident":
                confident_entropies.append(ent)
            elif gate == "uncertain":
                uncertain_entropies.append(ent)

    if not confident_entropies and not uncertain_entropies:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    if confident_entropies:
        ax.hist(confident_entropies, bins=20, alpha=0.6, label="Confident", color="steelblue")
    if uncertain_entropies:
        ax.hist(uncertain_entropies, bins=20, alpha=0.6, label="Uncertain", color="coral")
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Count")
    ax.set_title("Uncertainty Distribution by Gate Result")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"uncertainty_gating.{fmt}"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 7: LLM change margin vs correctness
# ---------------------------------------------------------------------------


def _extract_margin_from_llm_decision(llm_decision: dict) -> float | None:
    """Extract margin = |best - second_best| from candidate_scores combined_score."""
    scores = llm_decision.get("candidate_scores", [])
    if len(scores) < 2:
        return None
    combined = [c.get("combined_score") for c in scores if "combined_score" in c]
    if len(combined) < 2:
        return None
    sorted_scores = sorted(combined, reverse=True)  # higher = better
    return abs(sorted_scores[0] - sorted_scores[1])


def plot_margin_vs_correctness(trace, output_dir: str, fmt: str):
    correct_margins = []
    incorrect_margins = []

    for st in _attr(trace, "sentences") or []:
        for sp in _attr(st, "spans") or []:
            ld = _attr(sp, "llm_decision")
            if not ld:
                continue
            changed = ld.get("changed_from_top1", False) if isinstance(ld, dict) else _attr(ld, "changed_from_top1", False)
            if not changed:
                continue
            margin = _extract_margin_from_llm_decision(ld)
            if margin is None:
                continue
            correct = ld.get("change_was_correct") if isinstance(ld, dict) else _attr(ld, "change_was_correct")
            if correct:
                correct_margins.append(margin)
            else:
                incorrect_margins.append(margin)

    if not correct_margins and not incorrect_margins:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    data = []
    labels = []
    if correct_margins:
        data.append(correct_margins)
        labels.append("Correct")
    if incorrect_margins:
        data.append(incorrect_margins)
        labels.append("Incorrect")
    ax.boxplot(data)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Combined-score margin")
    ax.set_title("LLM Change Margin vs Correctness")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"margin_vs_correctness.{fmt}"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 8: Runtime comparison
# ---------------------------------------------------------------------------


def plot_runtime_comparison(baselines: list, rag: dict, output_dir: str, fmt: str):
    methods = []
    times = []

    for entry in baselines:
        methods.append(entry["name"])
        times.append(entry.get("time_s", 0))

    if rag:
        methods.append("confusion_rag")
        times.append(rag.get("time_s", 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(methods))
    ax.bar(x, times, color=sns.color_palette("viridis", len(methods)))
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Time (s)")
    ax.set_title("Runtime Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"runtime_comparison.{fmt}"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 9: Retrieval payload size
# ---------------------------------------------------------------------------


def plot_retrieval_payload(trace, output_dir: str, fmt: str):
    doc_counts = []
    for st in _attr(trace, "sentences") or []:
        for sp in _attr(st, "spans") or []:
            ret = _attr(sp, "retrieval")
            if ret:
                docs = ret.get("retrieved_docs", []) if isinstance(ret, dict) else _attr(ret, "retrieved_docs") or []
                if docs:
                    doc_counts.append(len(docs))

    if not doc_counts:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(doc_counts, bins=range(min(doc_counts), max(doc_counts) + 2), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Docs retrieved per span")
    ax.set_ylabel("Count")
    ax.set_title("Retrieval Payload Size Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"retrieval_payload.{fmt}"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Metrics table (LaTeX)
# ---------------------------------------------------------------------------


def write_metrics_table(baselines: list, rag: dict, output_dir: str):
    rows = []

    for entry in baselines:
        ci = entry.get("wer_ci_95", [None, None])
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "—"
        rows.append({
            "method": entry["name"],
            "wer": entry["wer"],
            "wer_ci": ci_str,
            "cer": entry.get("cer"),
            "oracle_wer": entry.get("oracle_wer"),
            "gap_closed": entry.get("oracle_gap_closed"),
            "time_s": entry.get("time_s"),
        })

    if rag:
        ci = rag.get("wer_ci_95", [None, None])
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "—"
        rows.append({
            "method": "confusion_rag",
            "wer": rag["wer"],
            "wer_ci": ci_str,
            "cer": rag.get("cer"),
            "oracle_wer": rag.get("oracle_wer"),
            "gap_closed": rag.get("oracle_gap_closed"),
            "time_s": rag.get("time_s"),
        })

    # LaTeX
    tex_path = os.path.join(output_dir, "metrics_table.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Method comparison on smoke test (40 utterances).}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write("Method & WER & 95\\% CI & CER & Oracle WER & Gap Closed & Time (s) \\\\\n")
        f.write("\\hline\n")
        for r in rows:
            cer = f"{r['cer']:.4f}" if r.get("cer") is not None else "---"
            oracle = f"{r['oracle_wer']:.4f}" if r.get("oracle_wer") is not None else "---"
            gap = f"{r['gap_closed']:.3f}" if r.get("gap_closed") is not None else "---"
            time_s = f"{r['time_s']:.2f}" if r.get("time_s") is not None else "---"
            method_tex = str(r["method"]).replace("_", "\\_")
            f.write(f"{method_tex} & {r['wer']:.4f} & {r['wer_ci']} & {cer} & {oracle} & {gap} & {time_s} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    # ConfusionRAG-specific addendum
    if rag:
        with open(tex_path, "a") as f:
            f.write("\n% ConfusionRAG-specific metrics\n")
            f.write(f"% High-uncertainty WER: {rag.get('high_uncertainty_wer')}\n")
            f.write(f"% Rare-word WER: {rag.get('rare_word_wer')}\n")
            f.write(f"% Mode breakdown: {rag.get('mode_breakdown_wer')}\n")
            faith = rag.get("faithfulness", {}).get("total", {})
            f.write(f"% Faithfulness: {faith}\n")


# ---------------------------------------------------------------------------
# Qualitative examples
# ---------------------------------------------------------------------------


def write_qualitative_examples(trace, output_dir: str):
    examples = []
    for st in _attr(trace, "sentences") or []:
        if not _attr(st, "was_changed", False):
            continue
        correct = None
        for sp in _attr(st, "spans") or []:
            ld = _attr(sp, "llm_decision")
            if ld:
                changed = ld.get("changed_from_top1", False) if isinstance(ld, dict) else _attr(ld, "changed_from_top1", False)
                if changed:
                    correct = ld.get("change_was_correct") if isinstance(ld, dict) else _attr(ld, "change_was_correct")
                    break
        examples.append({
            "idx": _attr(st, "sentence_idx", 0),
            "ground_truth": _attr(st, "ground_truth", ""),
            "top1": _attr(st, "top1_hypothesis", ""),
            "final": _attr(st, "final_decoded", ""),
            "correct": correct,
        })

    if not examples:
        return

    md_path = os.path.join(output_dir, "qualitative_examples.md")
    with open(md_path, "w") as f:
        f.write("# Qualitative Examples: LLM Changes\n\n")
        f.write("| Idx | Ground Truth | Top-1 | Final | Correct? |\n")
        f.write("|-----|--------------|-------|-------|----------|\n")
        for ex in examples:
            corr = "Yes" if ex["correct"] else "No"
            f.write(f"| {ex['idx']} | {ex['ground_truth']} | {ex['top1']} | {ex['final']} | {corr} |\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures and metrics")
    parser.add_argument("--results_dir", default="results_smoke", help="Directory with baselines.json and confusion_rag.json")
    parser.add_argument("--traces_dir", default="traces_smoke", help="Directory with trace JSON files")
    parser.add_argument("--output_dir", default="figures", help="Output directory for figures")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    if not HAS_PLOTTING:
        print("matplotlib/seaborn not installed. Install with: pip install -e '.[paper]'")
        print("Skipping figures; will only generate metrics_table.tex and qualitative_examples.md")

    # Resolve paths relative to ConfusionRAG root
    root = Path(__file__).resolve().parent.parent
    results_dir = root / args.results_dir
    traces_dir = root / args.traces_dir
    output_dir = root / args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    baselines, rag, trace = load_data(str(results_dir), str(traces_dir))

    # Publication style (only when plotting)
    if HAS_PLOTTING:
        sns.set_style("whitegrid")
        plt.rcParams["font.size"] = 10

    # Figures requiring baselines + rag
    if baselines and HAS_PLOTTING:
        plot_method_comparison(baselines, rag, str(output_dir), args.format)
        plot_oracle_gap_closed(baselines, rag, str(output_dir), args.format)
        plot_runtime_comparison(baselines, rag, str(output_dir), args.format)

    # Figures requiring rag
    if rag:
        if HAS_PLOTTING:
            plot_wer_by_mode(rag, str(output_dir), args.format)
            plot_faithfulness_accuracy(rag, str(output_dir), args.format)
        write_metrics_table(baselines or [], rag, str(output_dir))

    # Figures requiring trace
    if trace:
        if HAS_PLOTTING:
            plot_decision_mode_distribution(trace, str(output_dir), args.format)
            plot_uncertainty_gating(trace, str(output_dir), args.format)
            plot_margin_vs_correctness(trace, str(output_dir), args.format)
            plot_retrieval_payload(trace, str(output_dir), args.format)
        write_qualitative_examples(trace, str(output_dir))

    print(f"Figures and tables written to {output_dir}")


if __name__ == "__main__":
    main()
