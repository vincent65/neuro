# ConfusionRAG Experiment Examples for Paper

This document provides concrete, trace-backed examples from our trials to illustrate how different experiment configurations affect decoding decisions. Use these for figures, tables, or qualitative discussion in the paper.

---

## 1. Experiment Types Overview

| Experiment | Key Params | Purpose |
|------------|------------|---------|
| **Default** | retrieval_top_k=5, session_memory=10, nbest_rescore | Full pipeline with BM25 retrieval |
| **Ablation (no retrieval)** | retrieval_top_k=0, session_memory=0 | Isolate LLM rescoring without retrieval |
| **Margin guard** | nbest_change_margin_threshold=0.8 | Block low-confidence switches away from top-1 |
| **POC pair (control vs treatment)** | Control: top_k=0; Treatment: top_k=2, quality_gap=0.6 | Paired comparison of retrieval impact |
| **Prompt-choice** | llm_mode=span_choice, OpenAI GPT | Per-span word choice with evidence (vs nbest rescore) |
| **Phonetic / confusion-memory** | Alternative retrieval channels | Prototype retrieval variants |

---

## 2. Example 1: "and i loved ths shining"

**Ground truth:** `and i loved ths shining`  
**Top-1 hypothesis:** `and i love the line`

Confusion spans: *love/loved* (confident), *the/this* (uncertain), *line/signing/shining/...* (uncertain).

### Decisions by experiment

| Experiment | Final output | Notes |
|------------|--------------|-------|
| **Default** (nbest_rescore + retrieval) | `and i love the line` | Kept top-1. Retrieval returned "I love this country", "The child's love and security", etc. N-best rescore did not switch. |
| **Prompt-choice** (span_choice + retrieval) | `and i love the signing` | Changed *line* → *signing* (closer to *shining*). Evidence: "I love this country too much.", "The child's love and security." |
| **Ablation (no retrieval)** | `and i love the line` | Same as default; no retrieval, nbest rescore kept top-1. |

### Retrieval query (default / prompt-choice)

```
Query: "and i love (the OR this) line"
Retrieved: "I love this country too much.", "The child's love and security.", ...
```

---

## 3. Example 2: "actually it makes sense to a certain extent"

**Ground truth:** `actually it makes sense to a certain extent`  
**Top-1 hypothesis:** `actually it makes sense to a certain extend`

### Decisions by experiment

| Experiment | Final output | Change | Correct? |
|------------|--------------|--------|----------|
| **Default** | `actually it makes sense to a certain extent` | extend → extent | ✓ |
| **Ablation (no retrieval)** | `actually it makes sense to a certain extent` | extend → extent | ✓ |
| **Margin 0.8** | `actually it makes sense to a certain extent` | extend → extent | ✓ |

All runs corrected this. The N-best list contained the correct candidate; LLM rescoring selected it regardless of retrieval.

---

## 4. Example 3: Margin guard blocking a harmful change

**Ground truth:** `the measures already passed a senate committee`  
**Top-1 hypothesis:** `the measure was passed a senate committee`

N-best candidates include: *was passed*, *already passed*, *always passed*, *always placed*, etc.

### Decisions by experiment

| Experiment | Selected | raw_best_index | best_vs_second_margin | change_blocked_by_margin_guard |
|------------|----------|----------------|------------------------|-------------------------------|
| **Default** (margin=0) | `the measure already passed a senate committee` | 1 | — | false |
| **Ablation (no retrieval)** | `the measure always passed a senate committee` | 4 | 0.66 | false |
| **Margin 0.8** | `the measure was passed a senate committee` | 4 (raw) | 0.66 | **true** |

**Interpretation:** With margin=0.8, the LLM’s raw best was index 4 (*always passed*), but the margin over the second-best (0.66) was below 0.8. The guard forced retention of top-1 (*was passed*), avoiding the incorrect switch to *always passed*. The default run (with retrieval) had selected the correct *already passed*.

---

## 5. Example 4: Retrieval vs no retrieval — "in over will never allow..."

**Ground truth:** `char onion over grill never allow oil to smoke`  
**Top-1 hypothesis:** `in over will never allow all to see`

### Decisions by experiment

| Experiment | Final output | Change correct? |
|------------|--------------|-----------------|
| **Default** (retrieval) | `still in over will never allow all to see` | ✗ (wrong change) |
| **Ablation (no retrieval)** | `in over will never allow oil to smoke` | ✓ (oil, smoke correct) |

Without retrieval, the LLM chose *oil to smoke* from the N-best list, matching the ground-truth tail. With retrieval, it chose *still ... all to see*, which was incorrect. This illustrates retrieval sometimes introducing noise that steers the model away from the right candidate.

---

## 6. Example 5: POC pair — control vs treatment

**Best pair (pair_007):** control WER=0.1628, treatment WER=0.1512 (Δ = −0.0116).

- **Control:** `retrieval_top_k=0`, `session_memory=0`
- **Treatment:** `retrieval_top_k=2`, `evidence_max_docs=2`, `retrieval_quality_min_score_gap=0.6`
- Shared: `llm_alpha=0.65`, `acoustic_scale=0.5`, `llm_mode=nbest_rescore`

On the same chunk, retrieval-enabled treatment beat the no-retrieval control. Other pairs (e.g. pair_002, pair_003) showed retrieval hurting WER, so the effect is parameter-dependent.

---

## 7. Example 6: Prompt-choice span-by-span decisions

**Ground truth:** `so we're going to wait`  
**Top-1 hypothesis:** `they were going to write`

Prompt-choice operates per uncertain span. For the first word:

- **Candidates:** *they* (top-1), *so*, *we*, ...
- **Evidence docs:** (from retrieval)
- **Selected:** `so` ✓

**Another example — harmful change:**

**Ground truth:** `which i use as a reference and has been real handy`  
**Top-1 hypothesis:** `what i do as a wife and have been real handy`

Prompt-choice changed *what*→*eye*, *do*→*do*, *wife*→*wife*, etc. The change to *eye* was incorrect and illustrates retrieval/LLM noise in span_choice mode.

---

## 8. Summary Table: Key Param Differences

| Param | Default | Ablation noret | Margin 0.8 | POC treatment | Prompt-choice |
|-------|---------|----------------|------------|---------------|---------------|
| retrieval_top_k | 5 | 0 | 5 | 2 | 2 |
| session_memory_size | 10 | 0 | 10 | 0 | 0 |
| nbest_change_margin | 0 | 0 | 0.8 | 0 | — |
| llm_mode | nbest_rescore | nbest_rescore | nbest_rescore | nbest_rescore | span_choice |
| total_llm_changes (smoke) | 5 | 6 | 2 | 9 | 12 |
| faithfulness (smoke) | 0.40 | 0.50 | 1.0 | 0.56 | 0.25 |

---

## 9. Trace Locations

- Default: `traces_smoke_plan_20260226_default/`
- Ablation noret: `traces_smoke_plan_20260226_ablation_noret/`
- Margin 0.8: `traces_smoke_plan_20260226_margin_0.8/`
- POC pair 007: `traces_poc_pair_smoke_multi/pair_007_gpt2_control/`, `pair_007_gpt2_treatment/`
- Prompt-choice: `traces_prompt_choice_smoke/`

---

## 10. Suggested Paper Usage

1. **Table:** Example 3 (margin guard) — show how the guard blocks low-confidence changes.
2. **Table:** Example 4 — contrast retrieval vs no-retrieval on the same utterance.
3. **Figure:** Example 1 — confusion set, retrieval query, and outputs for default vs prompt-choice.
4. **Narrative:** Example 5 — POC pair as evidence that retrieval can help under the right params.
5. **Ablation:** Example 2 — both default and no-retrieval correct the same span, showing retrieval is not always necessary for fixes.
