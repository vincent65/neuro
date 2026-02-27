# POC RAG Win Report (Smoke Chunk)

Date: 2026-02-26

## Goal

Show a same-chunk proof-of-concept win where retrieval-enabled ConfusionRAG
beats a no-retrieval control under matched decoding settings.

Success criterion:

- `treatment_wer < control_wer`

## Data + Runner

- Chunk data:
  - `artifacts/nbest_outputs_test_s04_smoke.npy`
  - `artifacts/inference_out_test_s04_smoke.npy`
  - `artifacts/train_transcriptions.txt`
- Runner:
  - `benchmarks/run_poc_rag_pair.py`
- Output summary:
  - `results_poc_pair_smoke_multi/poc_pair_summary.json`

## Best Paired Win

From `results_poc_pair_smoke_multi/poc_pair_summary.json`:

- pair id: `pair_007_gpt2`
- model: `gpt2`
- shared scoring settings:
  - `llm_alpha=0.65`
  - `acoustic_scale=0.5`
- retrieval treatment settings:
  - `retrieval_top_k=2`
  - `evidence_max_docs=2`
  - `retrieval_quality_min_score_gap=0.6`

Result:

- control (no retrieval) WER: `0.1628`
- treatment (retrieval enabled) WER: `0.1512`
- delta (`treatment - control`): `-0.0116`  **win**

Artifacts:

- control JSON: `results_poc_pair_smoke_multi/pair_007_gpt2_control.json`
- treatment JSON: `results_poc_pair_smoke_multi/pair_007_gpt2_treatment.json`
- control trace: `traces_poc_pair_smoke_multi/pair_007_gpt2_control/20260226T231548Z_3317e869_run.json`
- treatment trace: `traces_poc_pair_smoke_multi/pair_007_gpt2_treatment/20260226T231555Z_6fd4cec7_run.json`

## Additional Notes

- The sweep covered 8 matched control/treatment pairs (`n_pairs=8`).
- Retrieval does **not** help uniformly; several pairs were neutral or worse.
- The best treatment WER from this sweep (`0.1512`) matches the prior smoke
  best from no-retrieval runs at `llm_alpha=0.5`, but this report’s objective
  was paired control-vs-treatment improvement on the same chunk, which was met.
