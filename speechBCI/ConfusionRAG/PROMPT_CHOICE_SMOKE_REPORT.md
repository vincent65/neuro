# Prompt-Choice RAG Smoke Report

Date: 2026-02-26

## Goal

Run a single prompt-choice benchmark on the smoke chunk and compare against existing saved results.

## Run Setup

- Runner: `benchmarks/run_prompt_choice_rag.py`
- Inputs:
  - `artifacts/nbest_outputs_test_s04_smoke.npy`
  - `artifacts/inference_out_test_s04_smoke.npy`
  - `artifacts/train_transcriptions.txt`
- Prompt model arg: `gpt-5`
- Retrieval/gating config:
  - `gate_metric=entropy`
  - `gate_threshold=0.5`
  - `retrieval_top_k=2`
  - `evidence_max_docs=2`
  - `retrieval_quality_gate_enabled=true`
  - `retrieval_quality_min_score_gap=0.6`
  - `retrieval_quality_min_nonzero_docs=1`
  - `session_memory_size=0`

Important runtime note:

- `OPENAI_API_KEY` was not set in this environment during execution.
- The run completed in deterministic fallback mode for prompt-choice calls.

## New Run Outputs

- JSON: `results_prompt_choice_smoke/prompt_choice_smoke_gpt5.json`
- Trace: `traces_prompt_choice_smoke/20260226T234812Z_52cf4c42_run.json`
- Comparison table: `results_prompt_choice_smoke/prompt_choice_smoke_gpt5_comparison.md`

Main metrics:

- WER: `0.170543`
- CER: `0.106383`
- Top-1 WER: `0.166667`
- High-uncertainty WER: `0.250000`
- Faithfulness (`prompt_choice`): `0.250` (3 correct / 12 changes)

## Comparison vs Existing Saved Results

From:

- `results_smoke/baselines.json`
- `results_smoke/confusion_rag.json`
- `results_poc_pair_smoke/pair_000_gpt2_control.json`

Key deltas (`new_run - baseline`):

- vs `ngram_top1`: `+0.003876` WER (worse)
- vs `llm_rescore_no_rag`: `+0.023256` WER (worse)
- vs `always_on_rag`: `+0.015504` WER (worse)
- vs `context_only`: `+0.015504` WER (worse)
- vs `confusion_rag.json`: `+0.011628` WER (worse)
- vs `pair_000_gpt2_control.json`: `+0.019380` WER (worse)

## Interpretation

For this smoke run, prompt-choice did not improve over existing baselines. In this environment, the result is expected to underperform because OpenAI calls were not active (`OPENAI_API_KEY` missing), so decisions used deterministic fallback behavior instead of live GPT-5 selection.
