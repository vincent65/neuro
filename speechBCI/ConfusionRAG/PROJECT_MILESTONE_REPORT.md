# CS224N Project Milestone: Confusion-Set Guided Retrieval for LLM-Constrained Brain-to-Text Decoding

**Repository:** neuro (extends speechBCI neural speech prosthesis)  
**Date:** February 2026

---

## Team

*[To be filled: List the members of your team (names and SUIDs)]*

---

## Mentor

*[To be filled: List your mentor]*

---

## Problem Description

We are investigating **brain-to-text decoding** for neural speech prostheses. The Willett et al. (2023) Nature system converts neural signals to phoneme logits via a GRU RNN with CTC, then decodes words using an N-gram language model and optionally rescores N-best hypotheses with an LLM. A key challenge is that the decoder often produces **confusable word pairs** (e.g., "green" vs. "greene" vs. "cream") where the neural evidence is ambiguous.

**Our problem:** Can we improve decoding accuracy by (1) identifying where the N-best hypotheses disagree (confusion spans), (2) retrieving relevant evidence only when the decoder is uncertain, and (3) using a constrained LLM to select among neural-evidence-supported candidates—thereby avoiding hallucination?

**Key constraints:** The LLM must never produce words outside the N-best list; every output must be traceable to the decoder's hypotheses. We add an uncertainty-gated retrieval-augmented generation (RAG) layer as a post-decoding step, without modifying the upstream speechBCI code.

---

## Data

### Dataset

We use the **speechBCI** dataset from the Willett et al. (2023) Nature paper: neural recordings from a participant with amyotrophic lateral sclerosis (ALS) attempting to speak, paired with ground-truth transcriptions. The data is organized by session; we use a **smoke split** of 40 utterances from Session 4 for rapid iteration.

### Data Collection and Preparation

Data flows through the upstream speechBCI pipeline:

1. **Neural features** → GRU RNN (CTC) → **phoneme logits**
2. **Phoneme logits** + N-gram LM → **N-best hypotheses** via `lmDecoderUtils.nbest_with_lm_decoder()`
3. **Inference output** from `NeuralSequenceDecoder.inference()` contains ground-truth transcriptions

**Required artifacts** (saved as `.npy` or `.txt`):

| Artifact | Format | Source |
|----------|--------|--------|
| `nbest_outputs.npy` | `List[List[(sentence, ac_score, lm_score)]]` per utterance | Upstream `5gram+llm_rescoring.ipynb` or equivalent |
| `inference_out.npy` | Dict with `transcriptions` (null-terminated byte arrays), `logits`, `logitLengths`, etc. | Upstream `rnn_step3_baselineRNNInference.ipynb` |
| `train_transcriptions.txt` | One sentence per line | Training transcriptions for BM25 retrieval corpus |

**Smoke chunk paths** (used in our experiments):

- `speechBCI/ConfusionRAG/artifacts/nbest_outputs_test_s04_smoke.npy`
- `speechBCI/ConfusionRAG/artifacts/inference_out_test_s04_smoke.npy`
- `speechBCI/ConfusionRAG/artifacts/train_transcriptions.txt`

**Data split:** The smoke split contains 40 utterances. Ground truth is extracted from `inference_out["transcriptions"]` (null-terminated byte arrays converted to strings via `chr()`). No test leakage: the retrieval corpus uses training transcriptions only.

---

## Baseline

We implement **four baselines** plus our main ConfusionRAG pipeline.

### Baseline 1: N-gram Top-1 (`ngram_top1`)

Take the first hypothesis from each N-best list. Apply the same hypothesis cleaning as speechBCI: strip `>`, collapse double spaces, remove space before `,`, `.`, `?`. No LLM, no retrieval.

### Baseline 2: LLM Rescore, No RAG (`llm_rescore_no_rag`)

LLM N-best rescoring without retrieval. Uses `gate_threshold=0.0` so every sentence is rescored. Combined score: `alpha * llm_score + (1-alpha) * old_lm_score + acoustic_scale * acoustic_score`. **This is our best-performing baseline** on the smoke chunk (WER 0.1473).

### Baseline 3: Always-On RAG (`always_on_rag`)

Retrieval on every sentence (no uncertainty gating). BM25 over training transcriptions, top-k docs per sentence, session memory appended. Same LLM rescoring as Baseline 2 but with evidence prefix.

### Baseline 4: Context-Only (`context_only`)

Prior decoded sentences as LLM context, no retrieval corpus, no confusion-set gating. Tests whether session continuity alone helps.

### ConfusionRAG (Our Method)

1. **Confusion-set construction:** Align N-best hypotheses at the word level (Levenshtein DP), extract disagreement spans with candidates and weights.
2. **Uncertainty gating:** Compute entropy, margin, or disagreement_mass per span; flag spans as uncertain if metric exceeds threshold.
3. **Retrieval:** For uncertain spans only, build OR-expanded BM25 query, retrieve top-k docs, optionally append session memory.
4. **Constrained LLM decode:** Either (a) **span-choice**: replace each uncertain word by scoring confusion candidates in context, or (b) **nbest_rescore**: score full hypotheses with evidence prefix, combine with acoustic/old-LM scores.
5. **Trace emission:** Full JSON observability for every decision.

**Implementation:** All baselines and ConfusionRAG are in `speechBCI/ConfusionRAG/`. Upstream `speechBCI/` is unmodified.

---

## Evaluation Methodology

### Primary Metrics

- **WER (Word Error Rate):** Levenshtein edit distance at word level, normalized by reference word count. Primary comparison metric.
- **CER (Character Error Rate):** Same at character level.
- **WER 95% CI:** Bootstrap confidence interval (10,000 resamples) for statistical comparison.

### Secondary Metrics

- **Oracle WER:** Best achievable WER if we always pick the N-best hypothesis closest to the reference. Upper bound on improvement.
- **Oracle gap closed:** `(top1_wer - method_wer) / (top1_wer - oracle_wer)`. Fraction of the top-1→oracle gap recovered.
- **High-uncertainty WER:** WER over sentences that had at least one uncertain span (where ConfusionRAG invoked the LLM).
- **Rare-word WER:** WER over sentences containing words with corpus frequency below threshold (default 5).
- **Faithfulness:** Accuracy of LLM changes vs. ground truth, broken down by mode (span_choice, nbest_rescore).
- **Mode breakdown WER:** WER by decision mode (kept_top1, span_choice, nbest_rescore).

### Evaluation Pipeline

`full_evaluation()` in `confusionrag/eval.py` produces a report dict with all metrics. It is called automatically by `run_baselines.py` and `run_confusion_rag.py` after each run.

### Reproducibility

All runs write traces to JSON. Config snapshot, per-sentence decisions, retrieval queries, and LLM scores are logged. Load with `RunTrace.load(path)`.

---

## Results

### Baseline Performance (Smoke Chunk, 40 utterances)

| Method | WER | CER | Oracle WER | Oracle Gap Closed |
|--------|-----|-----|------------|-------------------|
| ngram_top1 | 0.1667 | 0.1056 | 0.0426 | 0.0 |
| **llm_rescore_no_rag** | **0.1473** | 0.0883 | 0.0426 | 0.156 |
| always_on_rag | 0.1550 | 0.0930 | 0.0426 | 0.094 |
| context_only | 0.1550 | 0.0898 | 0.0426 | 0.094 |
| ConfusionRAG (default) | 0.1589 | 0.0961 | 0.0426 | 0.062 |

**Best baseline:** `llm_rescore_no_rag` (WER 0.1473). ConfusionRAG default is worse by +0.0116 WER.

### ConfusionRAG Ablation and Tuning (Smoke Plan 2026-02-26)

We ran extensive sweeps:

| Stage | Config | Best WER |
|-------|--------|----------|
| Default | retrieval_top_k=5, session_memory=10 | 0.1589 |
| No retrieval | retrieval_top_k=0, session_memory=0 | **0.1512** |
| Margin guard | nbest_change_margin_threshold ∈ {0.3, 0.5, 0.8, 1.0} | 0.1589 (no WER gain; faithfulness ↑ at 0.8) |
| Score tuning | llm_alpha × acoustic_scale grid | 0.1589 |
| Gate sweep | gate_threshold ∈ {0.4, 0.5, 0.6, 0.7} | 0.1589–0.1628 |

**Best ConfusionRAG:** `ablation_noret` (retrieval and session memory disabled) — WER 0.1512. Still does not beat baseline (0.1473).

### Retrieval Prototypes

- **Confusion-memory channel:** WER 0.1589
- **Phonetic (Soundex) channel:** WER 0.1550 (better than confusion-memory)

### Key Findings

1. **Retrieval is net harmful** on this smoke chunk. Disabling it gave the largest improvement.
2. **Gating targets hard cases** (17/40 sentences had uncertain spans), but rescoring those cases does not reliably fix them (nbest_rescore WER 0.2241 vs. kept_top1 WER 0.1056).
3. **LLM change faithfulness** was low (40% accuracy in default run; 50% in no-retrieval ablation).
4. **Retrieval quality audit:** 52% of retrieval events had flat top-score separation; 64.6% of retrieved docs had zero BM25 score. Query OR-expansion often produced 6–20+ candidates.

---

## Appendix A: Detailed Method Descriptions (Reproducibility)

### A.1 Confusion Set Construction (`confusionrag/confusion_set.py`)

**Algorithm:**

1. **Hypothesis cleaning** (must match speechBCI): strip `>`, collapse `"  "` to `" "`, remove space before `,`, `.`, `?`.
2. **Hypothesis-level weights:** `combined = ac_score + lm_score`; softmax over hypotheses.
3. **Word alignment:** Levenshtein DP with backtracking between top-1 and each other hypothesis. Returns `(ref_word|None, hyp_word|None)` pairs.
4. **Per-position candidates:** For each word position in top-1, accumulate candidate words and weights from aligned hypotheses.
5. **Confusion spans:** At positions with >1 distinct candidate: `entropy = -Σ w·log(w)`, `margin = top - second`, `disagreement_mass = 1 - max(weights)`.
6. **Gating:** Span uncertain if `entropy >= threshold` (or `margin <= threshold`, or `disagreement_mass >= threshold` depending on `gate_metric`).

**Function:** `build_confusion_sets(nbest, gate_metric, gate_threshold, min_nbest=2)`.

### A.2 Retrieval (`confusionrag/retriever.py`)

- **Index:** BM25Okapi over tokenized corpus (lowercased, split on spaces).
- **Query:** `"... left_ctx (cand1 OR cand2 OR ...) right_ctx ..."` with `retrieval_context_window` words on each side.
- **Pruning:** `retrieval_max_query_candidates` (default 6), `retrieval_min_candidate_weight` (default 0.02).
- **Session memory:** Deque of prior decoded sentences, appended to retrieval results when slots available.

### A.3 Constrained LLM Decoding (`confusionrag/constrained_llm.py`)

**Span-choice:** For each uncertain span, substitute each candidate, prepend evidence, score with LLM, pick argmax.

**N-best rescore:** Score each full hypothesis with `evidence_prefix + hypothesis`. Combined: `alpha * llm + (1-alpha) * old_lm + acoustic_scale * acoustic`. Optional margin guard: only switch from top-1 if `best - second >= nbest_change_margin_threshold`.

**Evidence prefix format:**
```
Context:
- doc1
- doc2

```

### A.4 Configuration Reference (`confusionrag/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| gate_metric | "entropy" | entropy / margin / disagreement_mass |
| gate_threshold | 0.5 | Gating threshold |
| retrieval_top_k | 5 | BM25 docs per span (0 = no retrieval) |
| session_memory_size | 10 | Prior sentences (0 = disabled) |
| llm_mode | "nbest_rescore" | span_choice / nbest_rescore |
| llm_alpha | 0.5 | LLM vs. old LM weight |
| acoustic_scale | 0.5 | Acoustic score weight |
| nbest_change_margin_threshold | 0.0 | Min margin to switch from top-1 |

---

## Appendix B: Reproducibility Commands

### Environment

```bash
cd speechBCI/ConfusionRAG
pip install -e ".[test]"
# Python >= 3.9; deps: numpy, scipy, rank_bm25, transformers, torch, tabulate
```

### Run Baselines

```bash
python benchmarks/run_baselines.py \
  --nbest_path artifacts/nbest_outputs_test_s04_smoke.npy \
  --inf_path artifacts/inference_out_test_s04_smoke.npy \
  --corpus_path artifacts/train_transcriptions.txt \
  --output_dir results_smoke \
  --llm_name gpt2
```

### Run ConfusionRAG

```bash
python benchmarks/run_confusion_rag.py \
  --nbest_path artifacts/nbest_outputs_test_s04_smoke.npy \
  --inf_path artifacts/inference_out_test_s04_smoke.npy \
  --corpus_path artifacts/train_transcriptions.txt \
  --output_dir results_smoke \
  --trace_dir traces_smoke \
  --llm_name gpt2 \
  --llm_mode nbest_rescore \
  --gate_threshold 0.5
```

### Ablation (No Retrieval)

```bash
python benchmarks/run_confusion_rag.py \
  --nbest_path artifacts/nbest_outputs_test_s04_smoke.npy \
  --inf_path artifacts/inference_out_test_s04_smoke.npy \
  --corpus_path artifacts/train_transcriptions.txt \
  --output_dir results_ablation_noret \
  --allow_no_retrieval \
  --retrieval_top_k 0 \
  --session_memory_size 0
```

### Compare Results

```bash
python benchmarks/compare_results.py --results_dir results_smoke
```

### Run Tests

```bash
cd speechBCI/ConfusionRAG && python -m pytest tests/ -v
# 101 tests, all passing
```

---

## Appendix C: File Structure

```
speechBCI/ConfusionRAG/
├── confusionrag/
│   ├── config.py          # ConfusionRAGConfig
│   ├── confusion_set.py   # N-best alignment, confusion spans
│   ├── retriever.py       # BM25, session memory, query construction
│   ├── constrained_llm.py # span_choice, nbest_rescore
│   ├── pipeline.py        # decode_with_confusion_rag()
│   ├── tracer.py          # RunTrace, SentenceTrace, SpanTrace
│   └── eval.py            # full_evaluation()
├── tests/                 # 101 pytest tests
├── benchmarks/
│   ├── run_baselines.py
│   ├── run_confusion_rag.py
│   ├── compare_results.py
│   └── hf_utils.py
├── setup.py
├── CHANGELOG.md
├── AGENTS.md
├── CONFUSION_RAG_DIAGNOSIS.md
├── SMOKE_EXPERIMENT_FINDINGS_20260226.md
├── RETRIEVAL_QUALITY_AUDIT_20260226.md
└── RAG_RETRIEVAL_PROTOTYPE_RESULTS_20260226.md
```

---

## Appendix D: References

- Willett et al. (2023). *A high-performance speech neuroprosthesis*. Nature.
- speechBCI: `NeuralDecoder/neuralDecoder/utils/lmDecoderUtils.py` — N-best decoder, LLM rescoring.
- ConfusionRAG: `speechBCI/ConfusionRAG/CHANGELOG.md` — Per-file documentation and test list.
