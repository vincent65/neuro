# ConfusionRAG Smoke Experiment Findings (2026-02-26)

## Scope

This note summarizes the full implementation + tuning pass run against the smoke split, based on:

- `speechBCI/ConfusionRAG/artifacts/nbest_outputs_test_s04_smoke.npy`
- `speechBCI/ConfusionRAG/artifacts/inference_out_test_s04_smoke.npy`
- `speechBCI/ConfusionRAG/artifacts/train_transcriptions.txt`

Goal was to beat best baseline `llm_rescore_no_rag` (`WER=0.1473`) from:

- `speechBCI/ConfusionRAG/results_smoke/baselines.json`

## Initial Problem Statement

From the existing diagnosis and baseline files:

- Current default ConfusionRAG: `WER=0.1589`
- Best baseline (`llm_rescore_no_rag`): `WER=0.1473`
- Gap: `+0.0116` WER (ConfusionRAG worse)

Known issues from diagnosis:

- Retrieval/context noise likely hurting quality.
- Hard utterances were gated, but rescoring quality on those cases was poor.
- Faithfulness of changes was low.

## Code Changes Implemented

### 1) Conservative n-best change guard

Added a margin guard in n-best rescoring so we only switch away from top-1 when best-vs-second margin is strong enough.

Changed files:

- `speechBCI/ConfusionRAG/confusionrag/config.py`
  - Added `nbest_change_margin_threshold: float = 0.0`
- `speechBCI/ConfusionRAG/confusionrag/constrained_llm.py`
  - `nbest_rescore_decode(..., change_margin_threshold=0.0, ...)`
  - Guard logic:
    - if `best_idx != 0` and `(best_combined - second_best_combined) < threshold`, force `top1`.
  - Added trace fields:
    - `raw_best_index`
    - `best_vs_second_margin`
    - `change_margin_threshold`
    - `change_blocked_by_margin_guard`
- `speechBCI/ConfusionRAG/confusionrag/pipeline.py`
  - Passes config threshold into `nbest_rescore_decode`
- `speechBCI/ConfusionRAG/benchmarks/run_confusion_rag.py`
  - Added CLI args:
    - `--session_memory_size`
    - `--nbest_change_margin_threshold`

### 2) Tests added

Updated:

- `speechBCI/ConfusionRAG/tests/test_constrained_llm.py`

Added tests:

- guard blocks low-margin change
- guard allows high-margin change

Validation:

- Full suite passed: `107 passed`

## Experiment Plan Executed

All outputs were written to fresh directories (no overwrite).

### Stage 0: Baseline anchor

- Baselines reused from `results_smoke/baselines.json`
- Default ConfusionRAG rerun:
  - `results_smoke_plan_20260226_default/confusion_rag.json`
  - `WER=0.1589` (matches prior)

### Stage 1: Retrieval ablations

- No retrieval + no session memory:
  - `results_smoke_plan_20260226_ablation_noret/confusion_rag.json`
  - `WER=0.1512` (best achieved in this pass)
  - faithfulness accuracy `0.50`
- Minimal retrieval (`top_k=1`) + no session memory:
  - `results_smoke_plan_20260226_ablation_top1/confusion_rag.json`
  - `WER=0.1589` (no gain)

Interpretation: retrieval is likely net harmful on this smoke chunk.

### Stage 2: Margin guard sweep (`0.3, 0.5, 0.8, 1.0`)

Directories:

- `results_smoke_plan_20260226_margin_0.3`
- `results_smoke_plan_20260226_margin_0.5`
- `results_smoke_plan_20260226_margin_0.8`
- `results_smoke_plan_20260226_margin_1.0`

Result:

- WER remained `0.1589` for all thresholds in this setup.
- Faithfulness improved at high thresholds (`0.8/1.0` -> `accuracy=1.0`) but no WER gain.

### Stage 3: Score tuning (`llm_alpha x acoustic_scale`)

Grid:

- `llm_alpha in {0.35, 0.45, 0.55, 0.65}`
- `acoustic_scale in {0.3, 0.5, 0.7}`
- with: no retrieval, no session memory, margin threshold `0.8`

Best score-grid WER:

- `0.1589` (several ties, e.g. `a=0.55/ac=0.5`, `a=0.65/ac=0.5`)

No score-grid config beat Stage 1 no-retrieval baseline (`0.1512`).

### Stage 4: Gate retuning (`0.4, 0.5, 0.6, 0.7`)

Using best Stage 3 tie representative (`a=0.55/ac=0.5`, margin `0.8`, no retrieval/session):

- `results_smoke_plan_20260226_gate_0.4`: `WER=0.1589`
- `results_smoke_plan_20260226_gate_0.5`: `WER=0.1589`
- `results_smoke_plan_20260226_gate_0.6`: `WER=0.1628`
- `results_smoke_plan_20260226_gate_0.7`: `WER=0.1628`

Raising threshold reduced rescored count but worsened overall WER.

## Final Ranking (Key Runs)

1. **Best overall in this pass**
   - `results_smoke_plan_20260226_ablation_noret/confusion_rag.json`
   - `WER=0.1512`
2. High-faithfulness but no WER gain
   - `results_smoke_plan_20260226_margin_0.8/confusion_rag.json`
   - `WER=0.1589`, faithfulness `1.0`
3. Original default
   - `results_smoke_plan_20260226_default/confusion_rag.json`
   - `WER=0.1589`

Reference best baseline:

- `results_smoke/baselines.json` -> `llm_rescore_no_rag` with `WER=0.1473`

## Main Findings

1. **Retrieval remains the strongest negative factor** on this smoke chunk.
   - Disabling retrieval/session memory gave the largest improvement.
2. **Margin guard improves trustworthiness but not WER** (in tested range).
3. **Alpha/acoustic/gate sweeps had limited leverage** after removing retrieval.
4. **Current best ConfusionRAG result still does not beat baseline**.
   - Best achieved: `0.1512`
   - Target baseline: `0.1473`

## Recommended Next Steps for New Agent

1. **Model-swap ablation (high priority)**
   - Keep best current setup fixed:
     - `retrieval_top_k=0`
     - `session_memory_size=0`
     - start from `llm_alpha=0.5`, `acoustic_scale=0.5`
   - Compare GPT-2 vs stronger LLMs.
2. **Retune score interpolation after model swap**
   - Re-sweep `llm_alpha` per model; score scales differ by model.
3. **Keep margin guard instrumentation**
   - Use trace fields (`best_vs_second_margin`, guard-block flags) to calibrate thresholds.
4. **Do not prioritize retrieval tuning first**
   - On this chunk, retrieval consistently failed to help.

## Notes

- All code changes were confined to `speechBCI/ConfusionRAG/`.
- Upstream `speechBCI/` code was not modified.
- This run used smoke split only; conclusions should be validated on larger splits before finalizing.
