# ConfusionRAG Performance Diagnosis (Smoke Chunk, Session 4)

**Date:** 2026-02-26  
**Data:** 40 utterances from `nbest_outputs_test_s04_smoke.npy` / `inference_out_test_s04_smoke.npy`  
**Purpose:** Context for agents iterating on ConfusionRAG to beat baseline on one chunk.

---

## Baseline Quality (Confirmed Correct)

Inference pipeline fixes are working. Baseline metrics on this chunk:

| Method | WER | CER | Oracle gap closed |
|--------|-----|-----|-------------------|
| ngram_top1 | 0.1667 | 0.1056 | 0.0 |
| **llm_rescore_no_rag** | **0.1473** | 0.0883 | 0.156 |
| always_on_rag | 0.1550 | 0.0930 | 0.094 |
| context_only | 0.1550 | 0.0898 | 0.094 |
| **ConfusionRAG (default)** | 0.1589 | 0.0961 | 0.062 |

**Best baseline:** `llm_rescore_no_rag` (WER 0.1473). ConfusionRAG is worse by +0.0116 WER (~7.9% relative).

---

## Key Findings

### 1. Retrieval/Context Is Hurting More Than Helping

- `llm_rescore_no_rag` beats both `always_on_rag` and `context_only`.
- Adding retrieved docs or session-memory context degrades performance on this chunk.
- Evidence payload grows quickly: ~9–15 docs per rescored sentence (top_k=5 BM25 + session_memory_size=10).

### 2. Gating Targets Hard Cases, But Rescoring Fails Them

- 17/40 sentences have uncertain spans → nbest_rescore.
- 23/40 kept top-1 (no LLM invoked).
- **kept_top1 WER:** 0.1056  
- **nbest_rescore WER:** 0.2241  

The gate correctly flags difficult utterances, but rescoring those difficult ones does not reliably fix them.

### 3. LLM Change Faithfulness Is Low

- 5 sentence-level changes total.
- 2 correct, 3 incorrect → **40% accuracy**.
- Correct changes: idx 4 (extend→extent), idx 22 (was→already).
- Incorrect changes: idx 29 (to→two), idx 30 (added "still"), idx 38 (vacationed→vacation).

### 4. Evidence Payload Is Likely Over-Noisy

- Per-span retrieval returns top_k=5 BM25 docs.
- Session memory appends up to 10 prior decoded sentences.
- For nbest_rescore, all spans’ docs are merged → often 9–15 docs per sentence.
- Noisy or off-topic context may bias the LLM toward bad rewrites.

### 5. Score Margins Do Not Predict Correctness

- Correct changes: margins 0.90, 1.05.
- Incorrect changes: margins 0.09, 0.33, 1.38.
- A large combined-score margin does not guarantee a correct change (e.g. idx 30, margin 1.38).

---

## Artifact Paths

- **N-best:** `speechBCI/ConfusionRAG/artifacts/nbest_outputs_test_s04_smoke.npy`
- **Inference:** `speechBCI/ConfusionRAG/artifacts/inference_out_test_s04_smoke.npy`
- **Corpus:** `speechBCI/ConfusionRAG/artifacts/train_transcriptions.txt`
- **Trace:** `speechBCI/ConfusionRAG/traces_smoke/20260226T191049Z_d2ce843a_run.json`
- **Results:** `speechBCI/ConfusionRAG/results_smoke/baselines.json`, `confusion_rag.json`

---

## Recommended Directions for Iteration

1. **Reduce noisy context:** Lower `retrieval_top_k` (e.g. 1–2), or set `session_memory_size=0` to test without session memory.
2. **Make rescoring more conservative:** Only change from top-1 when combined-score margin exceeds a threshold, or when LLM strongly prefers a different hypothesis.
3. **Revisit gating:** Try `gate_threshold` higher (fewer rescored sentences) or lower (more rescored) to see if the gate is over/under-triggering.
4. **Tune llm_alpha:** Default 0.5; try 0.35–0.65 to shift weight between LLM and acoustic/old-LM scores.
5. **Ablation:** Run `gate_threshold=0` with `retrieval_top_k=1`, `session_memory_size=0` to isolate effect of minimal retrieval vs no retrieval.

---

## Config Used in Smoke Run

```python
gate_metric="entropy"
gate_threshold=0.5
retrieval_top_k=5
retrieval_context_window=5
session_memory_size=10
llm_mode="nbest_rescore"
llm_alpha=0.5
acoustic_scale=0.5
```

---

## Running Benchmarks on Smoke Chunk

```bash
cd speechBCI/ConfusionRAG

# Baselines
python benchmarks/run_baselines.py \
  --nbest_path artifacts/nbest_outputs_test_s04_smoke.npy \
  --inf_path artifacts/inference_out_test_s04_smoke.npy \
  --corpus_path artifacts/train_transcriptions.txt \
  --output_dir results_smoke

# ConfusionRAG
python benchmarks/run_confusion_rag.py \
  --nbest_path artifacts/nbest_outputs_test_s04_smoke.npy \
  --inf_path artifacts/inference_out_test_s04_smoke.npy \
  --corpus_path artifacts/train_transcriptions.txt \
  --output_dir results_smoke \
  --trace_dir traces_smoke
```
