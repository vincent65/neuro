# ConfusionRAG Implementation Changelog

This document records every file created or modified during the implementation of the Confusion-Set Guided Retrieval (RAG) + Constrained LLM Decoding pipeline, built on top of the speechBCI neural decoder. It is intended as a reference for future agents working on this codebase.

## Context

The speechBCI repository (cloned into `speechBCI/`) implements a brain-to-text decoder: a GRU RNN trained with CTC loss converts neural features to phoneme logits, then an n-gram language model decodes words, and optionally an LLM rescores the N-best list.

Our contribution (`speechBCI/ConfusionRAG/`) adds a post-decoding layer that:
1. Builds confusion sets from N-best hypothesis disagreements
2. Gates retrieval on uncertainty (only retrieves when the decoder is unsure)
3. Uses a constrained LLM to select among neural-evidence-supported candidates (no hallucination)

**No existing speechBCI files were modified.** All code is new and additive.

## Relationship to Existing Code

Our pipeline consumes outputs from two existing speechBCI functions and replaces one:

| Existing function | File | What it does | How we use it |
|---|---|---|---|
| `nbest_with_lm_decoder()` | `NeuralDecoder/neuralDecoder/utils/lmDecoderUtils.py:207-229` | Produces N-best lists of `(sentence, ac_score, lm_score)` tuples | We consume this as input to `build_confusion_sets()` |
| `rescore_with_gpt2()` | `NeuralDecoder/neuralDecoder/utils/lmDecoderUtils.py:37-62` | Scores hypotheses with an LLM | Our `_rescore_with_llm()` reimplements this with evidence conditioning |
| `cer_with_gpt2_decoder()` | `NeuralDecoder/neuralDecoder/utils/lmDecoderUtils.py:99-132` | Computes WER/CER after LLM rescoring | Our `pipeline.py` + `eval.py` replace this with richer metrics |

The inference output dict (`inferenceOut`) from `NeuralSequenceDecoder.inference()` has keys: `logits`, `logitLengths`, `transcriptions`, `trueSeqs`, `decodedSeqs`, `editDistances`, `trueSeqLengths`, `seqErrorRate`, `cer`.

---

## Files Created

### Package Scaffolding

#### `speechBCI/ConfusionRAG/setup.py` (21 lines)
- Package metadata and dependencies
- Core deps: `numpy`, `scipy`, `rank_bm25`, `transformers`, `torch`, `tabulate`
- Optional extras: `sentence-transformers` (dense retrieval), `pytest`/`pytest-cov` (testing)
- Python >= 3.9 required

#### `speechBCI/ConfusionRAG/confusionrag/__init__.py` (21 lines)
- Public API re-exports: `ConfusionRAGConfig`, `build_confusion_sets`, `ConfusionSpan`, `ConfusionSetResult`, `Retriever`, `span_choice_decode`, `nbest_rescore_decode`, `decode_with_confusion_rag`, `RunTrace`, `SentenceTrace`, `SpanTrace`

#### `speechBCI/ConfusionRAG/tests/__init__.py` (empty)
#### `speechBCI/ConfusionRAG/benchmarks/__init__.py` (empty)

---

### Core Modules

#### `speechBCI/ConfusionRAG/confusionrag/config.py` (46 lines)
- `ConfusionRAGConfig` dataclass with all pipeline hyperparameters
- **Confusion set**: `min_nbest` (default 2)
- **Gating**: `gate_metric` ("entropy"/"margin"/"disagreement_mass"), `gate_threshold` (0.5)
- **Retrieval**: `retrieval_top_k` (5), `retrieval_context_window` (5), `session_memory_size` (10)
- **LLM decoding**: `llm_mode` ("nbest_rescore"/"span_choice"), `llm_alpha` (0.5), `acoustic_scale` (0.5), `llm_length_penalty` (0.0)
- **Tracing**: `trace_enabled` (True), `trace_dir` ("./traces")
- Has `.to_dict()` for JSON serialization

#### `speechBCI/ConfusionRAG/confusionrag/confusion_set.py` (264 lines)
- **Data structures**: `ConfusionSpan` (span_start, span_end, candidates, weights, entropy, margin, disagreement_mass), `ConfusionSetResult` (top1_hypothesis, nbest_hypotheses, nbest_scores, spans, is_uncertain)
- **`_align_words(ref, hyp)`**: Levenshtein DP alignment with backtracking; returns `(ref_word|None, hyp_word|None)` pairs for matches, substitutions, insertions, deletions
- **Metric helpers**: `_softmax()`, `_entropy()`, `_margin()`, `_disagreement_mass()`
- **`build_confusion_sets(nbest, gate_metric, gate_threshold, min_nbest)`**: Main entry point. Cleans hypotheses, computes hypothesis-level softmax weights, aligns each hypothesis to top-1, collects per-position candidate words with accumulated weights, builds `ConfusionSpan` objects at disagreement positions, applies gating threshold

#### `speechBCI/ConfusionRAG/confusionrag/retriever.py` (183 lines)
- **`RetrievalResult`** dataclass: query, retrieved_docs, scores, retrieval_time_ms
- **`Retriever` class**:
  - Constructor builds BM25Okapi index from corpus (handles empty corpus gracefully)
  - `add_to_session_memory()` / `get_session_memory()` / `clear_session_memory()`: deque-based buffer of prior decoded sentences
  - `build_query(sentence_words, span)`: constructs OR-expansion query (e.g., `"at (green OR greene OR cream) library"`) with configurable context window
  - `retrieve(query, include_session_memory)`: BM25 scoring, top-k selection, appends session memory docs
  - `retrieve_for_span()`: convenience wrapper combining query construction + retrieval

#### `speechBCI/ConfusionRAG/confusionrag/constrained_llm.py` (330 lines)
- **`_rescore_with_llm(model, tokenizer, texts, length_penalty)`**: Scores text strings with causal LM (supports both TF and PyTorch models). Sums per-token log-probs minus length penalty. This reimplements `rescore_with_gpt2()` from speechBCI but works with arbitrary text (including evidence prefixes).
- **`_build_evidence_prefix(retrieved_docs, max_docs)`**: Formats retrieved docs as `"Context:\n- doc1\n- doc2\n"` prefix
- **`span_choice_decode(model, tokenizer, confusion_result, retrieval_results, sentence_trace, length_penalty)`** (Mode A):
  - Iterates over confusion spans
  - For uncertain spans: constructs candidate sentences by substituting each candidate word, prepends evidence prefix, scores all candidates with LLM, picks argmax
  - Records `SpanTrace` with `llm_decision.mode = "span_choice"`, per-candidate scores, selected candidate, changed_from_top1 flag
  - Sets `sentence_trace.decision_mode = "span_choice"`
- **`nbest_rescore_decode(model, tokenizer, confusion_result, retrieval_results, sentence_trace, alpha, acoustic_scale, length_penalty)`** (Mode B):
  - Merges all retrieved evidence into one prefix
  - Scores each full N-best hypothesis with `evidence_prefix + hypothesis`
  - Combines: `alpha * llm_score + (1-alpha) * old_lm_score + acoustic_scale * acoustic_score`
  - Records whole-sentence `SpanTrace` with `llm_decision.mode = "nbest_rescore"`, per-hypothesis score breakdown (llm_score, old_lm_score, acoustic_score, combined_score), selected_index
  - Also records per-span traces for disagreement visibility
  - Sets `sentence_trace.decision_mode = "nbest_rescore"`

#### `speechBCI/ConfusionRAG/confusionrag/pipeline.py` (190 lines)
- **`_extract_transcriptions(inference_out)`**: Extracts ground-truth strings from the inference output dict (same logic as `_extract_transcriptions` in speechBCI's lmDecoderUtils.py)
- **`decode_with_confusion_rag(nbest_outputs, inference_out, llm_model, llm_tokenizer, retriever, config)`**: Main pipeline entry point.
  - Creates `RunTrace` with config snapshot
  - For each utterance:
    1. Builds confusion sets via `build_confusion_sets()`
    2. Retrieves evidence for uncertain spans via `retriever.retrieve_for_span()`
    3. Routes to `span_choice_decode()` or `nbest_rescore_decode()` based on `config.llm_mode`; if no uncertain spans, keeps top-1 with `decision_mode = "kept_top1"`
    4. Populates correctness flags from ground truth if available
    5. Adds decoded sentence to session memory
  - Writes trace JSON if `config.trace_enabled`
  - Returns `{decoded_transcripts, confidences, trace_path, trace}`

#### `speechBCI/ConfusionRAG/confusionrag/tracer.py` (196 lines)
- **`SpanTrace`** dataclass: span_start, span_end, top1_word, confusion_candidates, uncertainty_metrics, gate_result ("uncertain"/"confident"), gate_threshold_used, retrieval (dict or None), llm_decision (dict with `mode` field or None), time_ms
- **`SentenceTrace`** dataclass: sentence_idx, ground_truth, top1_hypothesis, final_decoded, was_changed, decision_mode ("kept_top1"/"span_choice"/"nbest_rescore"), n_uncertain_spans, spans list, total_time_ms. Has `start_timer()`/`stop_timer()` helpers.
- **`RunSummary`** dataclass: total_sentences, sentences_with_uncertain_spans, sentences_using_span_choice, sentences_using_nbest_rescore, sentences_kept_top1, total_spans_evaluated, total_spans_gated_uncertain, total_retrievals, total_llm_changes, llm_change_accuracy
- **`RunTrace`** dataclass: run_id (UUID), timestamp (ISO 8601), config dict, summary, sentences list. Methods: `compute_summary()` (aggregates from sentences), `to_dict()`, `save(trace_dir)` (writes JSON), `load(path)` (class method, reconstructs from JSON)

#### `speechBCI/ConfusionRAG/confusionrag/eval.py` (315 lines)
- **Core metrics**: `levenshtein()`, `compute_wer()`, `compute_cer()`, `compute_wer_with_ci()` (bootstrap 95% CI with 10k resamples)
- **Oracle metrics**: `compute_oracle_wer()` (best achievable from N-best), `oracle_gap_closed()` (fraction of top1-to-oracle gap recovered)
- **Slice metrics**: `compute_slice_wer()` (WER over boolean-masked subset), `high_uncertainty_mask()` (from trace), `rare_word_mask()` (from word frequency), `build_word_freq()`
- **Faithfulness audit**: `faithfulness_audit(trace)` — counts LLM changes broken down by mode (span_choice/nbest_rescore), computes accuracy of changes
- **Mode breakdown**: `mode_breakdown_wer(trace)` — WER separated by decision_mode
- **`full_evaluation(decoded, reference, nbest_outputs, trace, corpus)`**: Produces comprehensive report dict with all metrics

---

### Benchmark Scripts

#### `speechBCI/ConfusionRAG/benchmarks/run_baselines.py` (227 lines)
- CLI script with argparse: `--nbest_path`, `--inf_path`, `--corpus_path`, `--output_dir`, `--llm_name`
- Runs 4 baselines:
  1. `baseline_ngram_top1()`: Top-1 from N-gram decoder (no LLM)
  2. `baseline_llm_rescore_no_rag()`: LLM N-best rescoring without retrieval (gate_threshold=0.0, retriever=None)
  3. `baseline_always_on_rag()`: Retrieval on every sentence, no uncertainty gating (gate_threshold=0.0, retriever=Retriever)
  4. `baseline_context_only()`: Prior sentences as context, no confusion sets (gate_threshold=100.0, nothing triggered)
- Each baseline computes WER, CER, Oracle WER, Oracle gap closed, timing
- Saves results to `{output_dir}/baselines.json`

#### `speechBCI/ConfusionRAG/benchmarks/run_confusion_rag.py` (144 lines)
- CLI script with full hyperparameter control: `--llm_mode`, `--gate_metric`, `--gate_threshold`, `--llm_alpha`, `--acoustic_scale`, `--retrieval_top_k`
- Builds `ConfusionRAGConfig`, `Retriever`, loads LLM, runs `decode_with_confusion_rag()`
- Runs `full_evaluation()` and saves to `{output_dir}/confusion_rag.json`
- Prints detailed results table

#### `speechBCI/ConfusionRAG/benchmarks/compare_results.py` (107 lines)
- Loads `baselines.json` and `confusion_rag.json`
- Prints comparison table (uses `tabulate` if available, plain-text fallback)
- Prints detailed confusion-RAG info: high-uncertainty WER, rare-word WER, mode breakdown, faithfulness stats, summary counts

---

### Test Files (101 tests total, all passing)

#### `speechBCI/ConfusionRAG/tests/test_confusion_set.py` (178 lines)
- `TestAlignWords`: identical, substitution, insertion, deletion, empty ref/hyp (6 tests)
- `TestMetrics`: softmax sums to one, entropy uniform/deterministic, margin, disagreement mass (6 tests)
- `TestBuildConfusionSets`: basic output type, finds disagreement spans at correct positions, candidates include all variants, weights sum to one, entropy/margin/disagreement_mass gating, high threshold blocks gating, empty/single/identical hypotheses, span to_dict (12 tests)

#### `speechBCI/ConfusionRAG/tests/test_retriever.py` (138 lines)
- `TestRetrieverInit`: creates index, empty corpus handles gracefully (2 tests)
- `TestQueryConstruction`: OR expansion, single candidate no OR, context window (3 tests)
- `TestRetrieval`: returns RetrievalResult, top_k limit, relevant doc ranked high, retrieve_for_span (4 tests)
- `TestSessionMemory`: add and get, memory limit, session memory in retrieval, clear memory, to_dict (5 tests)

#### `speechBCI/ConfusionRAG/tests/test_constrained_llm.py` (240 lines)
- Uses `unittest.mock.patch` to mock `_rescore_with_llm` with fixed scores
- `TestBuildEvidencePrefix`: empty, formats docs, max_docs limit (3 tests)
- `TestSpanChoice`: selects best candidate, trace records mode="span_choice", output only from confusion set, changed_from_top1 flag (4 tests)
- `TestNbestRescore`: selects best hypothesis, trace records mode="nbest_rescore", combined scoring formula, empty hypotheses, candidate_scores in trace (5 tests)

#### `speechBCI/ConfusionRAG/tests/test_pipeline.py` (195 lines)
- Uses synthetic N-best generators with controlled disagreements at positions 1 and 2
- `TestPipelineIntegration`: basic output structure, with retriever, span_choice mode, trace file written and loadable, high threshold keeps top-1, session memory grows, empty nbest (7 tests)

#### `speechBCI/ConfusionRAG/tests/test_eval.py` (241 lines)
- `TestLevenshtein`: identical, substitution, insertion, deletion, empty (5 tests)
- `TestWER`/`TestCER`: perfect, all wrong, partial (5 tests)
- `TestWERWithCI`: returns three values (1 test)
- `TestOracleWER`: oracle picks best, oracle worse than all (2 tests)
- `TestOracleGapClosed`: perfect recovery, no improvement, partial, zero gap (4 tests)
- `TestSliceMetrics`: slice WER, empty mask, high uncertainty mask, rare word mask, build word freq (5 tests)
- `TestFaithfulnessAudit`: total changes, per-mode counts, accuracy, empty trace (4 tests)
- `TestModeBreakdownWER`: basic, no ground truth (2 tests)

#### `speechBCI/ConfusionRAG/tests/test_tracer.py` (222 lines)
- `TestSpanTrace`: to_dict, defaults (2 tests)
- `TestSentenceTrace`: timer, to_dict excludes _start_time, spans in dict (3 tests)
- `TestRunSummary`: total sentences, mode counts, uncertain counts, retrieval count, change counts, to_dict (6 tests)
- `TestRunTraceSerialization`: roundtrip, summary in saved file, mode tags always present, empty trace saves, trace with error graceful (5 tests)

---

## Data Flow Summary

```
NeuralSequenceDecoder.inference()
    |
    v
inferenceOut {logits, logitLengths, transcriptions, ...}
    |
    v
nbest_with_lm_decoder()        [existing speechBCI code]
    |
    v
nbest_outputs: List[List[(sentence, ac_score, lm_score)]]
    |
    v
decode_with_confusion_rag()     [our pipeline.py]
    |
    |---> build_confusion_sets()         [confusion_set.py]
    |         Aligns N-best, finds disagreement spans,
    |         computes entropy/margin/disagreement_mass,
    |         applies gating threshold
    |
    |---> retriever.retrieve_for_span()  [retriever.py]
    |         BM25 query with OR-expansion,
    |         only for spans gated as "uncertain"
    |
    |---> span_choice_decode()  OR       [constrained_llm.py]
    |     nbest_rescore_decode()
    |         LLM scores candidates with evidence prefix,
    |         every decision logged to SentenceTrace/SpanTrace
    |
    |---> RunTrace.save()                [tracer.py]
    |
    v
{decoded_transcripts, confidences, trace_path, trace}
    |
    v
full_evaluation()               [eval.py]
    WER, CER, Oracle WER, Oracle gap closed,
    slice metrics, faithfulness audit, mode breakdown
```

## Key Decisions and Gotchas

1. **Empty corpus guard**: `Retriever.__init__` checks `if corpus` before creating BM25Okapi — an empty list would cause a ZeroDivisionError in `rank_bm25`.

2. **Mock pattern for tests**: Pipeline and constrained_llm tests use `unittest.mock.patch("confusionrag.constrained_llm._rescore_with_llm")` with a side_effect that returns `[0.0] * len(texts)`. The mock must return the correct number of scores matching the number of input texts, not a fixed large number.

3. **Hypothesis cleaning**: Both `build_confusion_sets()` and the existing speechBCI code clean hypotheses with the same replacements: strip `>`, collapse double spaces, remove space before `,`, `.`, `?`. This must stay in sync.

4. **Trace `_start_time` field**: `SentenceTrace` has a private `_start_time` field (used by `start_timer()`/`stop_timer()`). The `to_dict()` method explicitly pops it, and `RunTrace.load()` also strips it before reconstruction.

5. **Combined scoring formula** (N-best rescore mode): `alpha * new_llm_score + (1-alpha) * old_lm_score + acoustic_scale * acoustic_score`. This matches the structure in speechBCI's `gpt2_lm_decode()` but adds evidence conditioning on the LLM side.

6. **Ground truth extraction**: `_extract_transcriptions()` in `pipeline.py` mirrors the same function in speechBCI's `lmDecoderUtils.py` — it reads null-terminated byte arrays from `inference_out["transcriptions"]` and converts to strings.
