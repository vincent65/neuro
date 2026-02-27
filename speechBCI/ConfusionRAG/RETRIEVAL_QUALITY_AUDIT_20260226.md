# Retrieval Quality Audit (Smoke Trace)

Date: 2026-02-26

This audit quantifies retrieval noise signatures from the smoke traces:

- `traces_smoke_plan_20260226_default/20260226T195928Z_abe8396d_run.json`
- `traces_smoke_plan_20260226_ablation_noret/20260226T200327Z_b4d94ba1_run.json`

## Key Metrics (Default Retrieval Run)

- retrieval events: `25`
- query token length: mean `31.52`, median `17`, p90 `64`, max `135`
- queries with more than 50 tokens: `6/25`
- OR candidate width: median `6`, max `66`, count >20: `6/25`
- docs per retrieval event: mean `13.8`, median `15`, max `15`
- merged docs in n-best rescoring: mean `15.47`, median `15`, p90 `20`, max `23`
- retrieval score gap (`top1 - top2`): mean `0.80`, median `0.42`, min `0.00`
- flat-score retrieval events (`top1-top2 < 0.5`): `13/25` (`52.0%`)
- zero-score docs per retrieval: mean `8.92`, median `10`, max `13`
- zero-score doc ratio (all retrieved docs): `0.646`

## Comparison: Retrieval-Disabled Ablation

- retrieval events still logged by the pipeline: `25`
- docs per retrieval event: `0`
- merged docs in n-best rescoring: `0`

## Interpretation

1. Retrieval query construction frequently explodes candidate OR clauses.
2. N-best rescoring receives oversized evidence payloads (often 15+ docs).
3. A large fraction of appended docs have `0.0` score (low-value context).
4. Over half of retrieval events have flat top-score separation, suggesting weak retrieval confidence.

These measurements motivate:

- query candidate pruning,
- strict evidence caps,
- retrieval confidence gating,
- optional reranking before evidence injection.
