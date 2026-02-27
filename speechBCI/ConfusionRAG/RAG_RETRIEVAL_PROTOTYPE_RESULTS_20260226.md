# Retrieval Prototype Benchmark Results (Smoke)

Date: 2026-02-26

## References

- No-RAG baseline (`llm_rescore_no_rag`): `WER=0.1473`
  - Source: `results_smoke/baselines.json`
- Best prior ConfusionRAG run in smoke pass (retrieval off): `WER=0.1512`
  - Source: `results_smoke_plan_20260226_ablation_noret/confusion_rag.json`

## Shared Prototype Settings

All prototype runs used:

- `llm_mode=nbest_rescore`
- `gate_metric=entropy`
- `gate_threshold=0.5`
- `retrieval_top_k=2`
- `session_memory_size=0`
- `retrieval_max_query_candidates=4`
- `retrieval_min_candidate_weight=0.03`
- `evidence_max_docs=3`
- `retrieval_quality_gate_enabled=True`
- `retrieval_quality_min_score_gap=0.6`

## Run 1: Confusion-Memory Retrieval Channel

- Output: `results_smoke_plan_20260226_confusion_memory_proto/confusion_rag.json`
- Trace: `traces_smoke_plan_20260226_confusion_memory_proto/20260226T214806Z_e043ec3e_run.json`
- Extra knobs:
  - `confusion_memory_enabled=True`
  - `confusion_memory_top_k=2`
  - `confusion_memory_window=2`

Metrics:

- `WER=0.1589`
- `CER=0.0977`
- `faithfulness.total.accuracy=0.50`
- `time_s=249.9`

## Run 2: Phonetic Retrieval Channel

- Output: `results_smoke_plan_20260226_phonetic_proto/confusion_rag.json`
- Trace: `traces_smoke_plan_20260226_phonetic_proto/20260226T215359Z_c5d36516_run.json`
- Extra knobs:
  - `phonetic_retrieval_enabled=True`
  - `phonetic_retrieval_top_k=2`

Metrics:

- `WER=0.1550`
- `CER=0.0930`
- `faithfulness.total.accuracy=0.60`
- `time_s=263.7`

## Comparison Summary

| Method | WER | Delta vs no-RAG (0.1473) | Delta vs best prior ConfusionRAG (0.1512) |
|---|---:|---:|---:|
| Confusion-memory prototype | 0.1589 | +0.0116 | +0.0077 |
| Phonetic prototype | 0.1550 | +0.0078 | +0.0038 |

## Takeaway

1. The phonetic channel improves over confusion-memory in this setup.
2. Neither prototype beats the no-RAG baseline or the best retrieval-disabled ConfusionRAG smoke run.
3. These channels may still be useful as selective augmentation, but not as currently configured for always-on uncertain-span retrieval.
