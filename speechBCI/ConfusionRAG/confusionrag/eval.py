"""
Extended evaluation metrics for confusion-set guided decoding.

Builds on the speechBCI WER/CER utilities and adds:
- Oracle WER (best achievable from the N-best list)
- Oracle gap closed
- Slice metrics (high-uncertainty, rare words)
- Faithfulness audit from trace files
- Mode-breakdown WER/CER
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from confusionrag.tracer import RunTrace


# ---------------------------------------------------------------------------
# Core WER / CER (self-contained, no speechBCI dependency)
# ---------------------------------------------------------------------------

def levenshtein(r: list, h: list) -> int:
    """Word- or character-level edit distance."""
    n, m = len(r), len(h)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if r[i - 1] == h[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])
    return int(dp[n, m])


def compute_wer(decoded: List[str], reference: List[str]) -> float:
    total_err = 0
    total_words = 0
    for dec, ref in zip(decoded, reference):
        d_words = dec.strip().split()
        r_words = ref.strip().split()
        total_err += levenshtein(r_words, d_words)
        total_words += len(r_words)
    return total_err / max(total_words, 1)


def compute_cer(decoded: List[str], reference: List[str]) -> float:
    total_err = 0
    total_chars = 0
    for dec, ref in zip(decoded, reference):
        total_err += levenshtein(list(ref), list(dec))
        total_chars += len(ref)
    return total_err / max(total_chars, 1)


def compute_wer_with_ci(
    decoded: List[str],
    reference: List[str],
    n_resamples: int = 10000,
) -> Tuple[float, float, float]:
    """WER with 95% bootstrap confidence interval."""
    word_errs = []
    word_counts = []
    for dec, ref in zip(decoded, reference):
        d_words = dec.strip().split()
        r_words = ref.strip().split()
        word_errs.append(levenshtein(r_words, d_words))
        word_counts.append(len(r_words))

    word_errs = np.array(word_errs)
    word_counts = np.array(word_counts)
    wer = np.sum(word_errs) / max(np.sum(word_counts), 1)

    resampled = np.zeros(n_resamples)
    n = len(word_errs)
    for i in range(n_resamples):
        idx = np.random.randint(0, n, size=n)
        resampled[i] = np.sum(word_errs[idx]) / max(np.sum(word_counts[idx]), 1)

    ci_low, ci_high = np.percentile(resampled, [2.5, 97.5])
    return float(wer), float(ci_low), float(ci_high)


# ---------------------------------------------------------------------------
# Oracle WER
# ---------------------------------------------------------------------------

def compute_oracle_wer(
    nbest_outputs: List[List[Tuple[str, float, float]]],
    reference: List[str],
) -> float:
    """
    Best achievable WER if we always pick the N-best hypothesis closest to
    the reference.
    """
    total_err = 0
    total_words = 0
    for nbest, ref in zip(nbest_outputs, reference):
        r_words = ref.strip().split()
        total_words += len(r_words)

        best_err = levenshtein(r_words, [])  # worst case: empty decode
        for sent, _, _ in nbest:
            h = sent.strip().replace(">", "").replace("  ", " ")
            h = h.replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
            err = levenshtein(r_words, h.split())
            if err < best_err:
                best_err = err
        total_err += best_err

    return total_err / max(total_words, 1)


def oracle_gap_closed(
    top1_wer: float,
    method_wer: float,
    oracle_wer: float,
) -> float:
    """Fraction of (top-1 -> oracle) improvement recovered by the method."""
    gap = top1_wer - oracle_wer
    if gap <= 0:
        return 0.0
    return (top1_wer - method_wer) / gap


# ---------------------------------------------------------------------------
# Slice metrics
# ---------------------------------------------------------------------------

def compute_slice_wer(
    decoded: List[str],
    reference: List[str],
    mask: List[bool],
) -> Optional[float]:
    """WER over a boolean-masked subset of sentences."""
    dec_sub = [d for d, m in zip(decoded, mask) if m]
    ref_sub = [r for r, m in zip(reference, mask) if m]
    if not dec_sub:
        return None
    return compute_wer(dec_sub, ref_sub)


def high_uncertainty_mask(trace: RunTrace) -> List[bool]:
    """Sentences that had at least one uncertain span."""
    return [st.n_uncertain_spans > 0 for st in trace.sentences]


def rare_word_mask(
    reference: List[str],
    word_freq: Dict[str, int],
    freq_threshold: int = 5,
) -> List[bool]:
    """Sentences containing at least one word below the frequency threshold."""
    mask = []
    for ref in reference:
        words = ref.strip().lower().split()
        has_rare = any(word_freq.get(w, 0) < freq_threshold for w in words)
        mask.append(has_rare)
    return mask


def build_word_freq(corpus: List[str]) -> Dict[str, int]:
    """Build a word frequency map from a corpus of sentences."""
    freq: Dict[str, int] = {}
    for sent in corpus:
        for w in sent.strip().lower().split():
            freq[w] = freq.get(w, 0) + 1
    return freq


# ---------------------------------------------------------------------------
# Faithfulness audit (from trace)
# ---------------------------------------------------------------------------

def faithfulness_audit(trace: RunTrace) -> Dict[str, Any]:
    """
    Analyse the trace to compute faithfulness metrics, broken down by LLM
    mode.
    """
    results: Dict[str, Dict[str, int]] = {
        "span_choice": {"changes": 0, "correct": 0, "incorrect": 0},
        "nbest_rescore": {"changes": 0, "correct": 0, "incorrect": 0},
        "total": {"changes": 0, "correct": 0, "incorrect": 0},
    }

    for st in trace.sentences:
        for sp in st.spans:
            if sp.llm_decision is None:
                continue
            if not sp.llm_decision.get("changed_from_top1", False):
                continue

            mode = sp.llm_decision.get("mode", "unknown")
            if mode not in results and mode != "total":
                results[mode] = {"changes": 0, "correct": 0, "incorrect": 0}
            bucket = results.get(mode, results["total"])

            bucket["changes"] += 1
            results["total"]["changes"] += 1

            correct = sp.llm_decision.get("change_was_correct")
            if correct is True:
                bucket["correct"] += 1
                results["total"]["correct"] += 1
            elif correct is False:
                bucket["incorrect"] += 1
                results["total"]["incorrect"] += 1

    for bucket in results.values():
        total = bucket["changes"]
        bucket["accuracy"] = (
            bucket["correct"] / total if total > 0 else 0.0
        )

    return results


# ---------------------------------------------------------------------------
# Mode-breakdown WER
# ---------------------------------------------------------------------------

def mode_breakdown_wer(trace: RunTrace) -> Dict[str, Optional[float]]:
    """
    WER broken down by decision_mode (kept_top1 / span_choice / nbest_rescore).
    """
    mode_dec: Dict[str, List[str]] = {}
    mode_ref: Dict[str, List[str]] = {}

    for st in trace.sentences:
        mode = st.decision_mode or "unknown"
        mode_dec.setdefault(mode, []).append(st.final_decoded)
        mode_ref.setdefault(mode, []).append(st.ground_truth)

    out: Dict[str, Optional[float]] = {}
    for mode in mode_dec:
        if not mode_ref[mode] or all(r == "" for r in mode_ref[mode]):
            out[mode] = None
        else:
            out[mode] = compute_wer(mode_dec[mode], mode_ref[mode])
    return out


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def full_evaluation(
    decoded: List[str],
    reference: List[str],
    nbest_outputs: List[List[Tuple[str, float, float]]],
    trace: RunTrace,
    corpus: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Produce a comprehensive evaluation report.

    Parameters
    ----------
    decoded : list of str
        Final decoded sentences from our pipeline.
    reference : list of str
        Ground-truth sentences.
    nbest_outputs : list of list
        N-best hypothesis lists per utterance.
    trace : RunTrace
        Trace from the pipeline run.
    corpus : list of str or None
        Training corpus for rare-word slicing.
    """
    wer, wer_lo, wer_hi = compute_wer_with_ci(decoded, reference)
    cer = compute_cer(decoded, reference)

    # Top-1 WER (first hypothesis from each N-best)
    top1 = []
    for nbest in nbest_outputs:
        if nbest:
            h = nbest[0][0].strip().replace(">", "").replace("  ", " ")
            h = h.replace(" ,", ",").replace(" .", ".").replace(" ?", "?")
            top1.append(h)
        else:
            top1.append("")
    top1_wer = compute_wer(top1, reference)

    oracle_w = compute_oracle_wer(nbest_outputs, reference)
    gap = oracle_gap_closed(top1_wer, wer, oracle_w)

    unc_mask = high_uncertainty_mask(trace)
    unc_wer = compute_slice_wer(decoded, reference, unc_mask)

    word_freq = build_word_freq(corpus) if corpus else {}
    rare_mask = rare_word_mask(reference, word_freq) if word_freq else []
    rare_wer = compute_slice_wer(decoded, reference, rare_mask) if rare_mask else None

    faith = faithfulness_audit(trace)
    mode_wer = mode_breakdown_wer(trace)

    return {
        "wer": wer,
        "wer_ci_95": [wer_lo, wer_hi],
        "cer": cer,
        "top1_wer": top1_wer,
        "oracle_wer": oracle_w,
        "oracle_gap_closed": gap,
        "high_uncertainty_wer": unc_wer,
        "rare_word_wer": rare_wer,
        "faithfulness": faith,
        "mode_breakdown_wer": mode_wer,
        "summary": trace.summary.to_dict(),
    }
