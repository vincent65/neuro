"""
End-to-end confusion-set guided RAG decoding pipeline.

Orchestrates:
1. Confusion-set construction from N-best outputs.
2. Uncertainty-gated retrieval.
3. Constrained LLM decoding (span-choice or N-best rescoring).
4. JSON trace emission for full observability.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from confusionrag.config import ConfusionRAGConfig
from confusionrag.confusion_set import ConfusionSetResult, build_confusion_sets
from confusionrag.constrained_llm import nbest_rescore_decode, span_choice_decode
from confusionrag.retriever import Retriever, RetrievalResult
from confusionrag.tracer import RunTrace, SentenceTrace, SpanTrace


def _word_error_count(reference: str, hypothesis: str) -> int:
    """Word-level edit distance for sentence-level correctness checks."""
    r_words = reference.strip().split()
    h_words = hypothesis.strip().split()
    n, m = len(r_words), len(h_words)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if r_words[i - 1] == h_words[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])
    return int(dp[n, m])


def _extract_transcriptions(inference_out: dict) -> List[str]:
    """Extract ground-truth transcription strings from inference output."""
    transcriptions: List[str] = []
    for i in range(len(inference_out["transcriptions"])):
        raw = inference_out["transcriptions"][i]
        end_idx = np.argwhere(raw == 0)
        if len(end_idx) == 0:
            end_idx = len(raw)
        else:
            end_idx = end_idx[0, 0]
        text = "".join(chr(c) for c in raw[:end_idx])
        transcriptions.append(text.strip())
    return transcriptions


def decode_with_confusion_rag(
    nbest_outputs: List[List],
    inference_out: dict,
    llm_model,
    llm_tokenizer,
    retriever: Optional[Retriever],
    config: ConfusionRAGConfig,
) -> Dict[str, Any]:
    """
    Run the full confusion-set guided RAG pipeline on a batch of utterances.

    Parameters
    ----------
    nbest_outputs : list of list
        Per-utterance N-best lists, each entry is a list of
        ``(sentence, acoustic_score, lm_score)`` tuples (as produced by
        ``lmDecoderUtils.nbest_with_lm_decoder``).
    inference_out : dict
        Inference output from ``NeuralSequenceDecoder.inference()``.
    llm_model, llm_tokenizer
        HuggingFace causal LM + tokenizer.
    retriever : Retriever or None
        If None, retrieval is skipped (useful for ablation baselines).
    config : ConfusionRAGConfig
        Pipeline configuration.

    Returns
    -------
    dict with keys:
        decoded_transcripts : list of str
        confidences : list of float
        trace_path : str or None
        trace : RunTrace
    """
    run_trace = RunTrace(config=config.to_dict())

    # Extract ground-truth if available
    try:
        true_transcriptions = _extract_transcriptions(inference_out)
    except (KeyError, TypeError):
        true_transcriptions = [""] * len(nbest_outputs)

    decoded_sentences: List[str] = []
    confidences: List[float] = []

    for utt_idx in range(len(nbest_outputs)):
        st = SentenceTrace(sentence_idx=utt_idx)
        st.start_timer()

        nbest = nbest_outputs[utt_idx]

        if utt_idx < len(true_transcriptions):
            st.ground_truth = true_transcriptions[utt_idx]

        # --- Step 1: Build confusion sets ---
        cs = build_confusion_sets(
            nbest,
            gate_metric=config.gate_metric,
            gate_threshold=config.gate_threshold,
            min_nbest=config.min_nbest,
        )
        st.top1_hypothesis = cs.top1_hypothesis
        st.n_uncertain_spans = sum(cs.is_uncertain) if cs.is_uncertain else 0

        # --- Step 2: Retrieve evidence for uncertain spans ---
        retrieval_results: Dict[int, RetrievalResult] = {}
        if retriever is not None:
            sentence_words = cs.top1_hypothesis.split()
            for span_idx, (span, uncertain) in enumerate(
                zip(cs.spans, cs.is_uncertain)
            ):
                if uncertain:
                    rr = retriever.retrieve_for_span(sentence_words, span)
                    retrieval_results[span_idx] = rr

        # --- Step 3: Constrained LLM decoding ---
        has_uncertain = st.n_uncertain_spans > 0
        has_hypotheses = len(cs.nbest_hypotheses) > 0

        if not has_hypotheses:
            decoded = ""
            st.decision_mode = "kept_top1"
        elif not has_uncertain:
            decoded = cs.top1_hypothesis
            st.decision_mode = "kept_top1"
        elif config.llm_mode == "span_choice":
            decoded = span_choice_decode(
                llm_model,
                llm_tokenizer,
                cs,
                retrieval_results,
                st,
                evidence_max_docs=config.evidence_max_docs,
                retrieval_quality_gate_enabled=config.retrieval_quality_gate_enabled,
                retrieval_quality_min_top_score=config.retrieval_quality_min_top_score,
                retrieval_quality_min_score_gap=config.retrieval_quality_min_score_gap,
                retrieval_quality_min_nonzero_docs=config.retrieval_quality_min_nonzero_docs,
                length_penalty=config.llm_length_penalty,
                llm_batch_size=config.llm_batch_size,
            )
        else:
            decoded = nbest_rescore_decode(
                llm_model,
                llm_tokenizer,
                cs,
                retrieval_results,
                st,
                alpha=config.llm_alpha,
                acoustic_scale=config.acoustic_scale,
                change_margin_threshold=config.nbest_change_margin_threshold,
                evidence_max_docs=config.evidence_max_docs,
                retrieval_quality_gate_enabled=config.retrieval_quality_gate_enabled,
                retrieval_quality_min_top_score=config.retrieval_quality_min_top_score,
                retrieval_quality_min_score_gap=config.retrieval_quality_min_score_gap,
                retrieval_quality_min_nonzero_docs=config.retrieval_quality_min_nonzero_docs,
                length_penalty=config.llm_length_penalty,
                llm_batch_size=config.llm_batch_size,
            )

        st.final_decoded = decoded
        st.was_changed = decoded != cs.top1_hypothesis

        # Confidence: probability of selected hypothesis under combined scores
        if len(cs.nbest_scores) > 1:
            combined = np.array([s[0] + s[1] for s in cs.nbest_scores])
            combined -= np.max(combined)
            probs = np.exp(combined) / np.sum(np.exp(combined))
            confidences.append(float(np.max(probs)))
        else:
            confidences.append(1.0)

        # Populate correctness flags if ground truth available
        if st.ground_truth:
            gt_words = st.ground_truth.split()
            for sp in st.spans:
                if sp.llm_decision is not None and sp.llm_decision.get("selected"):
                    mode = sp.llm_decision.get("mode")
                    selected = sp.llm_decision["selected"]
                    if mode == "nbest_rescore":
                        # Sentence-level decision: "correct" means strictly fewer
                        # word errors than the original top-1 sentence.
                        selected_err = _word_error_count(st.ground_truth, selected)
                        top1_err = _word_error_count(st.ground_truth, st.top1_hypothesis)
                        sp.llm_decision["change_was_correct"] = selected_err < top1_err
                    elif sp.span_start < len(gt_words):
                        gt_word = gt_words[sp.span_start]
                        sp.llm_decision["change_was_correct"] = (
                            selected.lower() == gt_word.lower()
                        )

        st.stop_timer()
        decoded_sentences.append(decoded)

        # Update session memory
        if retriever is not None:
            retriever.add_to_session_memory(decoded)

        run_trace.sentences.append(st)

    # --- Write trace ---
    trace_path = None
    if config.trace_enabled:
        trace_path = run_trace.save(config.trace_dir)

    return {
        "decoded_transcripts": decoded_sentences,
        "confidences": confidences,
        "trace_path": trace_path,
        "trace": run_trace,
    }
