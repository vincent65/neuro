"""
Build confusion sets and uncertainty signals from N-best hypothesis lists.

Given the N-best output of the n-gram LM decoder (list of (sentence, ac_score,
lm_score) tuples per utterance), this module:

1. Aligns hypotheses at the word level to find disagreement spans.
2. Extracts a confusion set for each span (distinct candidates + weights).
3. Computes uncertainty metrics (entropy, margin, disagreement mass).
4. Applies an uncertainty gate to decide which spans need retrieval.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConfusionSpan:
    """A word-level span where N-best hypotheses disagree."""
    span_start: int
    span_end: int
    candidates: List[str]
    weights: np.ndarray
    entropy: float = 0.0
    margin: float = 0.0
    disagreement_mass: float = 0.0

    def to_dict(self) -> dict:
        return {
            "span_start": self.span_start,
            "span_end": self.span_end,
            "candidates": self.candidates,
            "weights": self.weights.tolist(),
            "entropy": self.entropy,
            "margin": self.margin,
            "disagreement_mass": self.disagreement_mass,
        }


@dataclass
class ConfusionSetResult:
    """Confusion-set analysis for a single utterance."""
    top1_hypothesis: str
    nbest_hypotheses: List[str]
    nbest_scores: List[Tuple[float, float]]  # (ac_score, lm_score) per hyp
    spans: List[ConfusionSpan] = field(default_factory=list)
    is_uncertain: List[bool] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Word-level alignment via Levenshtein DP with back-tracking
# ---------------------------------------------------------------------------

def _align_words(ref: List[str], hyp: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Align two word sequences using edit-distance DP.

    Returns a list of (ref_word | None, hyp_word | None) pairs representing
    matches, substitutions, insertions, and deletions.
    """
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])

    # back-track
    alignment: List[Tuple[Optional[str], Optional[str]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            alignment.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            alignment.append((ref[i - 1], hyp[j - 1]))  # substitution
            i -= 1
            j -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            alignment.append((None, hyp[j - 1]))  # insertion
            j -= 1
        else:
            alignment.append((ref[i - 1], None))  # deletion
            i -= 1

    alignment.reverse()
    return alignment


# ---------------------------------------------------------------------------
# Core: build confusion sets from N-best
# ---------------------------------------------------------------------------

def _combined_score(ac: float, lm: float) -> float:
    return ac + lm


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def _entropy(weights: np.ndarray) -> float:
    w = weights[weights > 0]
    return float(-np.sum(w * np.log(w + 1e-12)))


def _margin(weights: np.ndarray) -> float:
    if len(weights) < 2:
        return float("inf")
    sorted_w = np.sort(weights)[::-1]
    return float(sorted_w[0] - sorted_w[1])


def _disagreement_mass(weights: np.ndarray) -> float:
    if len(weights) == 0:
        return 0.0
    return float(1.0 - np.max(weights))


def build_confusion_sets(
    nbest: List[Tuple[str, float, float]],
    gate_metric: str = "entropy",
    gate_threshold: float = 0.5,
    min_nbest: int = 2,
) -> ConfusionSetResult:
    """
    Analyse a single utterance's N-best list and produce confusion spans.

    Parameters
    ----------
    nbest : list of (sentence, acoustic_score, lm_score)
        N-best hypotheses from the LM decoder.
    gate_metric : str
        Which uncertainty metric to use for gating ("entropy", "margin",
        "disagreement_mass").
    gate_threshold : float
        Threshold for the gating decision.
    min_nbest : int
        Skip confusion-set analysis if fewer than this many hypotheses.

    Returns
    -------
    ConfusionSetResult
    """
    # Clean hypotheses
    hypotheses: List[str] = []
    scores: List[Tuple[float, float]] = []
    for sent, ac, lm in nbest:
        h = sent.strip()
        if len(h) == 0:
            continue
        h = h.replace(">", "").replace("  ", " ").replace(" ,", ",")
        h = h.replace(" .", ".").replace(" ?", "?")
        hypotheses.append(h)
        scores.append((ac, lm))

    if len(hypotheses) == 0:
        return ConfusionSetResult(
            top1_hypothesis="",
            nbest_hypotheses=[],
            nbest_scores=[],
        )

    top1 = hypotheses[0]
    result = ConfusionSetResult(
        top1_hypothesis=top1,
        nbest_hypotheses=hypotheses,
        nbest_scores=scores,
    )

    if len(hypotheses) < min_nbest:
        return result

    # Compute combined scores and hypothesis-level weights
    combined = np.array([_combined_score(ac, lm) for ac, lm in scores])
    hyp_weights = _softmax(combined)

    ref_words = top1.split()

    # Collect per-position candidate words by aligning each hypothesis to top-1
    # position_candidates[i] maps candidate_word -> accumulated weight
    position_candidates: Dict[int, Dict[str, float]] = {
        i: {} for i in range(len(ref_words))
    }
    # Record the top-1 candidate at every position
    for i, w in enumerate(ref_words):
        position_candidates[i][w] = position_candidates[i].get(w, 0.0) + hyp_weights[0]

    for h_idx in range(1, len(hypotheses)):
        hyp_words = hypotheses[h_idx].split()
        alignment = _align_words(ref_words, hyp_words)

        ref_pos = 0
        for ref_w, hyp_w in alignment:
            if ref_w is not None:
                token = hyp_w if hyp_w is not None else ""
                if ref_pos < len(ref_words):
                    d = position_candidates[ref_pos]
                    d[token] = d.get(token, 0.0) + hyp_weights[h_idx]
                ref_pos += 1
            # insertions (ref_w is None) are aggregated as extra text at the
            # preceding ref position — for simplicity we skip them in the
            # per-position confusion set and handle them only in full-hypothesis
            # N-best rescoring mode.

    # Build spans where top-1 word is not unanimously agreed upon
    spans: List[ConfusionSpan] = []
    is_uncertain: List[bool] = []
    for pos in range(len(ref_words)):
        cands = position_candidates[pos]
        # Remove empty-string candidate (deletion) for cleaner sets
        cands_clean = {k: v for k, v in cands.items() if k != ""}
        if len(cands_clean) <= 1:
            continue

        candidates = list(cands_clean.keys())
        raw_weights = np.array([cands_clean[c] for c in candidates])
        weights = raw_weights / (raw_weights.sum() + 1e-12)

        ent = _entropy(weights)
        mar = _margin(weights)
        dis = _disagreement_mass(weights)

        span = ConfusionSpan(
            span_start=pos,
            span_end=pos + 1,
            candidates=candidates,
            weights=weights,
            entropy=ent,
            margin=mar,
            disagreement_mass=dis,
        )
        spans.append(span)

        if gate_metric == "entropy":
            is_uncertain.append(ent >= gate_threshold)
        elif gate_metric == "margin":
            is_uncertain.append(mar <= gate_threshold)
        elif gate_metric == "disagreement_mass":
            is_uncertain.append(dis >= gate_threshold)
        else:
            is_uncertain.append(True)

    result.spans = spans
    result.is_uncertain = is_uncertain
    return result
