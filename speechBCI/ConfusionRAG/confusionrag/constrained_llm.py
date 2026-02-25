"""
Constrained LLM decoding: span-choice and evidence-conditioned N-best rescoring.

Both modes prevent hallucination by constraining the LLM to select among
candidates already supported by the neural decoder's N-best output.  Every
function records its decisions into a SentenceTrace / SpanTrace so the active
mode and all scoring details are always visible in the JSON trace output.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from confusionrag.confusion_set import ConfusionSetResult, ConfusionSpan
from confusionrag.retriever import RetrievalResult
from confusionrag.tracer import SentenceTrace, SpanTrace


# ---------------------------------------------------------------------------
# Helpers shared between modes
# ---------------------------------------------------------------------------

def _rescore_with_llm(
    model,
    tokenizer,
    texts: List[str],
    length_penalty: float = 0.0,
) -> List[float]:
    """
    Score each text string with the LLM.  Returns per-text log-prob scores
    (sum of token log-probs minus a length penalty).

    Supports both TF (TFGPT2LMHeadModel) and PyTorch causal LM models.
    """
    model_class = type(model).__name__

    if model_class.startswith("TF"):
        import tensorflow as tf
        inputs = tokenizer(texts, return_tensors="tf", padding=True)
        outputs = model(inputs)
        log_probs = tf.math.log(tf.nn.softmax(outputs["logits"], -1)).numpy()
    else:
        import torch
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            log_probs = torch.nn.functional.log_softmax(
                outputs["logits"].float(), -1
            ).cpu().numpy()

    scores: List[float] = []
    B, T, _ = log_probs.shape
    for i in range(B):
        attn = inputs["attention_mask"][i]
        if hasattr(attn, "numpy"):
            attn = attn.numpy()
        elif hasattr(attn, "cpu"):
            attn = attn.cpu().numpy()
        n_tokens = int(np.sum(attn))

        ids = inputs["input_ids"][i]
        if hasattr(ids, "numpy"):
            ids = ids.numpy()
        elif hasattr(ids, "cpu"):
            ids = ids.cpu().numpy()

        score = 0.0
        for j in range(1, n_tokens):
            score += log_probs[i, j - 1, ids[j]]
        scores.append(score - n_tokens * length_penalty)

    return scores


def _build_evidence_prefix(retrieved_docs: List[str], max_docs: int = 5) -> str:
    """Format retrieved documents into a prefix string for the LLM."""
    docs = retrieved_docs[:max_docs]
    if not docs:
        return ""
    lines = ["Context:"] + [f"- {d}" for d in docs] + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mode A: Span-choice
# ---------------------------------------------------------------------------

def span_choice_decode(
    model,
    tokenizer,
    confusion_result: ConfusionSetResult,
    retrieval_results: Dict[int, RetrievalResult],
    sentence_trace: SentenceTrace,
    length_penalty: float = 0.0,
) -> str:
    """
    Span-choice decoding: keep top-1 hypothesis fixed except at uncertain
    spans, where the LLM scores each confusion candidate in context and
    picks the best one.

    Parameters
    ----------
    model, tokenizer
        HuggingFace causal LM.
    confusion_result
        Output of ``build_confusion_sets`` for this utterance.
    retrieval_results
        Mapping from span index (into confusion_result.spans) to its
        RetrievalResult.  Spans not in this dict had gating = confident.
    sentence_trace
        Trace object for this sentence — decisions are written into it.
    length_penalty
        LLM length penalty.

    Returns
    -------
    str : the final decoded sentence.
    """
    words = confusion_result.top1_hypothesis.split()

    llm_applied = False
    for span_idx, span in enumerate(confusion_result.spans):
        t0 = time.perf_counter()
        st = SpanTrace(
            span_start=span.span_start,
            span_end=span.span_end,
            top1_word=words[span.span_start] if span.span_start < len(words) else "",
            confusion_candidates=[
                {"word": c, "weight": float(w)}
                for c, w in zip(span.candidates, span.weights)
            ],
            uncertainty_metrics={
                "entropy": span.entropy,
                "margin": span.margin,
                "disagreement_mass": span.disagreement_mass,
            },
        )

        is_uncertain = confusion_result.is_uncertain[span_idx]
        st.gate_result = "uncertain" if is_uncertain else "confident"

        if not is_uncertain or span_idx not in retrieval_results:
            st.gate_threshold_used = 0.0
            st.time_ms = (time.perf_counter() - t0) * 1000
            sentence_trace.spans.append(st)
            continue

        rr = retrieval_results[span_idx]
        st.retrieval = rr.to_dict()

        evidence_prefix = _build_evidence_prefix(rr.retrieved_docs)

        candidate_texts = []
        for cand in span.candidates:
            trial_words = list(words)
            trial_words[span.span_start : span.span_end] = [cand]
            full_sentence = " ".join(trial_words)
            candidate_texts.append(evidence_prefix + full_sentence)

        llm_scores = _rescore_with_llm(model, tokenizer, candidate_texts, length_penalty)
        best_idx = int(np.argmax(llm_scores))
        selected = span.candidates[best_idx]
        top1_word = words[span.span_start] if span.span_start < len(words) else ""

        words[span.span_start : span.span_end] = [selected]

        st.llm_decision = {
            "mode": "span_choice",
            "candidate_scores": [
                {"candidate": c, "score": float(s)}
                for c, s in zip(span.candidates, llm_scores)
            ],
            "selected": selected,
            "changed_from_top1": selected != top1_word,
            "change_was_correct": None,  # populated later if ground truth available
        }
        llm_applied = True
        st.time_ms = (time.perf_counter() - t0) * 1000
        sentence_trace.spans.append(st)

    sentence_trace.decision_mode = "span_choice" if llm_applied else "kept_top1"
    return " ".join(words)


# ---------------------------------------------------------------------------
# Mode B: N-best rescoring with evidence
# ---------------------------------------------------------------------------

def nbest_rescore_decode(
    model,
    tokenizer,
    confusion_result: ConfusionSetResult,
    retrieval_results: Dict[int, RetrievalResult],
    sentence_trace: SentenceTrace,
    alpha: float = 0.5,
    acoustic_scale: float = 0.5,
    length_penalty: float = 0.0,
) -> str:
    """
    N-best rescoring: score every hypothesis with the LLM conditioned on
    retrieved evidence, then combine with acoustic and old LM scores.

    Parameters
    ----------
    model, tokenizer
        HuggingFace causal LM.
    confusion_result
        Output of ``build_confusion_sets``.
    retrieval_results
        Mapping from span index to RetrievalResult.  All retrieved docs are
        merged into a single evidence prefix.
    sentence_trace
        Trace object.
    alpha, acoustic_scale, length_penalty
        Scoring hyperparameters (see plan).

    Returns
    -------
    str : the best hypothesis.
    """
    hypotheses = confusion_result.nbest_hypotheses
    scores = confusion_result.nbest_scores

    if not hypotheses:
        sentence_trace.decision_mode = "nbest_rescore"
        return ""

    # Merge all retrieved evidence into one prefix
    all_docs: List[str] = []
    seen: set = set()
    for rr in retrieval_results.values():
        for doc in rr.retrieved_docs:
            if doc not in seen:
                all_docs.append(doc)
                seen.add(doc)
    evidence_prefix = _build_evidence_prefix(all_docs)

    # Record a single "whole-sentence" span trace for N-best rescoring
    t0 = time.perf_counter()

    texts_for_llm = [evidence_prefix + h for h in hypotheses]
    llm_scores = _rescore_with_llm(model, tokenizer, texts_for_llm, length_penalty)

    acoustic_scores = np.array([s[0] for s in scores])
    old_lm_scores = np.array([s[1] for s in scores])
    new_lm_scores = np.array(llm_scores)

    combined = (
        alpha * new_lm_scores
        + (1 - alpha) * old_lm_scores
        + acoustic_scale * acoustic_scores
    )

    best_idx = int(np.argmax(combined))
    best_hyp = hypotheses[best_idx]
    top1_hyp = hypotheses[0]

    elapsed = (time.perf_counter() - t0) * 1000

    # Build span trace representing the whole-sentence decision
    st = SpanTrace(
        span_start=0,
        span_end=len(top1_hyp.split()),
        top1_word=top1_hyp,
        confusion_candidates=[
            {"word": h, "weight": float(w)}
            for h, w in zip(hypotheses, combined)
        ],
        uncertainty_metrics={},
        gate_result="uncertain",
        retrieval={
            "query": "(merged evidence for n-best rescoring)",
            "retrieved_docs": all_docs,
            "retrieval_time_ms": sum(
                rr.retrieval_time_ms for rr in retrieval_results.values()
            ),
        } if retrieval_results else None,
        llm_decision={
            "mode": "nbest_rescore",
            "candidate_scores": [
                {
                    "candidate": h,
                    "llm_score": float(ls),
                    "old_lm_score": float(ol),
                    "acoustic_score": float(ac),
                    "combined_score": float(cs),
                }
                for h, ls, ol, ac, cs in zip(
                    hypotheses, llm_scores, old_lm_scores,
                    acoustic_scores, combined,
                )
            ],
            "selected": best_hyp,
            "selected_index": best_idx,
            "changed_from_top1": best_idx != 0,
            "change_was_correct": None,
        },
        time_ms=elapsed,
    )

    # Also record per-disagreement-span traces for visibility
    for span_idx, span in enumerate(confusion_result.spans):
        is_uncertain = confusion_result.is_uncertain[span_idx]
        words = top1_hyp.split()
        sub_st = SpanTrace(
            span_start=span.span_start,
            span_end=span.span_end,
            top1_word=words[span.span_start] if span.span_start < len(words) else "",
            confusion_candidates=[
                {"word": c, "weight": float(w)}
                for c, w in zip(span.candidates, span.weights)
            ],
            uncertainty_metrics={
                "entropy": span.entropy,
                "margin": span.margin,
                "disagreement_mass": span.disagreement_mass,
            },
            gate_result="uncertain" if is_uncertain else "confident",
        )
        if span_idx in retrieval_results:
            sub_st.retrieval = retrieval_results[span_idx].to_dict()
        sentence_trace.spans.append(sub_st)

    sentence_trace.spans.append(st)
    sentence_trace.decision_mode = "nbest_rescore"
    return best_hyp
