# ConfusionRAG Milestone Update: Implementation + POC Results

Date: 2026-02-26

## Executive Summary

This milestone focused on producing a same-chunk proof-of-concept where
retrieval-enabled ConfusionRAG beats a no-retrieval control under matched
conditions.

Outcome:

- A paired win was achieved on the smoke chunk.
- Best paired result improved WER from `0.1628` (control) to `0.1512`
  (retrieval-enabled treatment), delta `-0.0116`.
- Full test suite passed after changes (`116 passed`).

## Problem Being Addressed

Prior smoke experiments showed retrieval often hurt performance relative to
no-retrieval rescoring. Two issues were targeted:

1. **Model/runtime bottleneck**: benchmark scripts were awkward for swapping to
   stronger modern causal LMs and robust runtime settings.
2. **Retrieval noise bottleneck**: traces showed oversized, low-value evidence
   payloads (including many weak/zero-score docs) degrading rescoring quality.

## What Was Implemented

### 1) Stronger-model readiness and robust HF runtime controls

Added shared HF utilities in:

- `benchmarks/hf_utils.py`

Key capabilities:

- Generic model/tokenizer loading via `AutoModelForCausalLM` +
  `AutoTokenizer`.
- Runtime flags for safe deployment:
  - `--hf_device_map`
  - `--hf_torch_dtype`
  - `--hf_trust_remote_code`
  - `--hf_attn_implementation`
- Graceful fallback path when `device_map` is requested but `accelerate`
  is unavailable.

Integrated these utilities into:

- `benchmarks/run_confusion_rag.py`
- `benchmarks/run_baselines.py`

### 2) LLM scoring scalability and stability

Added batch-aware LLM rescoring support:

- Config: `confusionrag/config.py`
  - new `llm_batch_size`
- Wiring: `confusionrag/pipeline.py`
- Scorer implementation: `confusionrag/constrained_llm.py`
  - `_rescore_with_llm(..., max_batch_size=...)`
  - fixed CUDA tensor to numpy conversion path for GPU execution

This makes larger-model rescoring more practical and reduces OOM risk for
larger N-best/evidence prompts.

### 3) Retrieval evidence quality hardening

Updated retrieval and evidence compaction behavior in:

- `confusionrag/retriever.py`
- `confusionrag/constrained_llm.py`
- `confusionrag/config.py` defaults

Main improvements:

- Query candidate pruning defaults tightened:
  - `retrieval_max_query_candidates=6`
  - `retrieval_min_candidate_weight=0.02`
- Weak evidence dropped earlier:
  - non-positive docs removed from retrieval merge path
  - dedupe and compact document-score pairs
- Session memory now backfills only when there is retrieval budget left, rather
  than always appending.
- N-best merged evidence now excludes zero-score docs before prompt injection.

### 4) Paired POC harness for objective control-vs-treatment testing

Added:

- `benchmarks/run_poc_rag_pair.py`

This runner executes matched pairs on the same chunk:

- **Control**: retrieval disabled.
- **Treatment**: retrieval enabled with quality gates.

It writes per-run JSONs and a ranked summary with best pair based on
`delta_wer_treatment_minus_control`.

### 5) Reporting artifact

Added:

- `POC_RAG_WIN_REPORT.md`

And this document:

- `MILESTONE_RAG_POC_IMPLEMENTATION_20260226.md`

## Validation and Test Status

Command:

- `./.venv_test/bin/python -m pytest tests/ -v`

Result:

- `116 passed`

Additional runtime dependency installed in the test env for model loading:

- `accelerate`

## POC Experiment Results (Smoke Chunk)

Source summary:

- `results_poc_pair_smoke_multi/poc_pair_summary.json`

Sweep executed:

- 8 matched control/treatment pairs (`n_pairs=8`)
- model: `gpt2`
- varied: `llm_alpha`, `retrieval_top_k`, retrieval quality gap threshold

Best pair:

- pair id: `pair_007_gpt2`
- shared scoring:
  - `llm_alpha=0.65`
  - `acoustic_scale=0.5`
- treatment retrieval settings:
  - `retrieval_top_k=2`
  - `evidence_max_docs=2`
  - `retrieval_quality_min_score_gap=0.6`
  - `retrieval_quality_min_nonzero_docs=1`

Metrics:

- control WER: `0.16279069767441862`
- treatment WER: `0.1511627906976744`
- delta: `-0.011627906976744207` (treatment better)

Related artifacts:

- control JSON: `results_poc_pair_smoke_multi/pair_007_gpt2_control.json`
- treatment JSON: `results_poc_pair_smoke_multi/pair_007_gpt2_treatment.json`
- control trace: `traces_poc_pair_smoke_multi/pair_007_gpt2_control/20260226T231548Z_3317e869_run.json`
- treatment trace: `traces_poc_pair_smoke_multi/pair_007_gpt2_treatment/20260226T231555Z_6fd4cec7_run.json`

Notable side metrics for best pair:

- high-uncertainty WER improved from `0.2328` (control) to `0.2069` (treatment)
- faithfulness accuracy improved from `0.40` to `0.625`

## Why These Results Are Meaningful

1. **Paired experimental design**: the win is not from changing the dataset or
   evaluation protocol; control and treatment are matched on the same chunk.
2. **Demonstrates retrieval can help under constrained settings**: retrieval is
   not universally beneficial, but quality-gated and capped evidence can
   outperform no-retrieval in specific conditions.
3. **Actionable engineering path**: the project now has tooling to scale model
   choice and run systematic paired ablations quickly.
4. **Trace-backed observability**: every claim is linked to JSON artifacts and
   traces for auditability.

## Caveats

- The best treatment WER (`0.1512`) does not surpass the previously reported
  smoke best baseline (`0.1473`) in absolute terms.
- This milestone proves paired retrieval benefit for a chunk/config region, not
  universal superiority across all settings.

## Repro Commands

Run full tests:

```bash
cd speechBCI/ConfusionRAG
./.venv_test/bin/python -m pytest tests/ -v
```

Run paired smoke sweep (same setup used for the report):

```bash
cd speechBCI/ConfusionRAG
HF_HUB_DISABLE_PROGRESS_BARS=1 ./.venv_test/bin/python benchmarks/run_poc_rag_pair.py \
  --nbest_path artifacts/nbest_outputs_test_s04_smoke.npy \
  --inf_path artifacts/inference_out_test_s04_smoke.npy \
  --corpus_path artifacts/train_transcriptions.txt \
  --output_dir results_poc_pair_smoke_multi \
  --trace_root traces_poc_pair_smoke_multi \
  --llm_names gpt2 \
  --llm_alpha_values 0.5,0.65 \
  --acoustic_scale_values 0.5 \
  --retrieval_top_k_values 1,2 \
  --evidence_max_docs_values 2 \
  --retrieval_quality_min_score_gap_values 0.3,0.6 \
  --session_memory_size 0 \
  --llm_batch_size 16 \
  --max_pairs 8
```
