# Agent Instructions for neuro/

This repository implements a CS224N project: **Confusion-Set Guided Retrieval for LLM-Constrained Brain-to-Text Decoding**. It extends the speechBCI neural speech prosthesis with an uncertainty-gated RAG + constrained LLM post-decoding layer.

## Repository Layout

```
neuro/
├── speechBCI/                          # Upstream (DO NOT MODIFY)
│   ├── NeuralDecoder/                  # GRU RNN + CTC decoder (TensorFlow)
│   │   ├── neuralDecoder/
│   │   │   ├── models.py              # GRU model class
│   │   │   ├── neuralSequenceDecoder.py  # Training + inference
│   │   │   ├── datasets/speechDataset.py # TFRecord data pipeline
│   │   │   ├── utils/
│   │   │   │   ├── lmDecoderUtils.py  # N-gram + LLM decoding (KEY INTERFACE)
│   │   │   │   └── rnnEval.py         # WER via Levenshtein distance
│   │   │   └── configs/               # Hydra YAML configs
│   │   └── setup.py
│   ├── LanguageModelDecoder/           # Kaldi-style FST decoder (C++)
│   ├── AnalysisExamples/               # Jupyter notebooks for data prep + eval
│   └── ...
├── speechBCI/ConfusionRAG/             # OUR CODE (editable)
│   ├── confusionrag/                   # Main package
│   │   ├── config.py                   # ConfusionRAGConfig dataclass
│   │   ├── confusion_set.py           # N-best alignment + confusion spans
│   │   ├── retriever.py               # BM25 retrieval + session memory
│   │   ├── constrained_llm.py         # Span-choice + N-best rescore modes
│   │   ├── pipeline.py                # End-to-end orchestrator
│   │   ├── tracer.py                  # JSON trace file emission
│   │   └── eval.py                    # WER/CER + oracle + faithfulness
│   ├── tests/                          # 101 pytest tests (all passing)
│   ├── benchmarks/                     # CLI scripts for baselines + comparison
│   ├── setup.py
│   └── CHANGELOG.md                    # Detailed per-file documentation
└── AGENTS.md                           # This file
```

## Critical Rules

### Do NOT modify speechBCI/

The `speechBCI/` directory is upstream code from the Willett et al. (2023) Nature paper. It must remain untouched. All our work lives under `speechBCI/ConfusionRAG/`. The only interaction with upstream code is consuming its outputs as inputs to our pipeline.

### Always run tests after changes

```bash
cd speechBCI/ConfusionRAG && python -m pytest tests/ -v
```

There are 101 tests. All must pass. If you introduce a test failure, fix it before finishing.

### Hypothesis cleaning must stay in sync

Both `build_confusion_sets()` (in `confusion_set.py`) and the upstream speechBCI code apply identical hypothesis cleaning: strip `>`, collapse `"  "` to `" "`, remove space before `,`, `.`, `?`. If you change one, you must change the other or results will silently diverge.

### Mock pattern for LLM in tests

Tests mock `_rescore_with_llm` via `unittest.mock.patch("confusionrag.constrained_llm._rescore_with_llm")`. The mock's return value **must** match the number of input texts (use `[0.0] * len(texts)`, not a fixed count). A mismatch causes broadcast errors in numpy.

## Architecture

The system is a post-processing layer that sits after the existing neural decoder pipeline:

```
Neural Features
  --> GRU RNN (CTC)        [speechBCI, frozen]
  --> Phoneme Logits
  --> N-gram LM (N-best)   [speechBCI, frozen]
  --> ConfusionRAG          [our code]
      1. build_confusion_sets()   -- align N-best, find disagreement spans
      2. Uncertainty gate          -- entropy/margin/disagreement_mass threshold
      3. BM25 retrieval            -- only for uncertain spans
      4. Constrained LLM decode   -- span_choice OR nbest_rescore mode
      5. Trace emission            -- full JSON observability
  --> Final text + evaluation
```

### Two LLM Decoding Modes

**Span-choice** (`config.llm_mode = "span_choice"`): Keeps top-1 hypothesis fixed, replaces individual uncertain words by scoring each confusion candidate in context with the LLM. Traces tag each decision with `mode: "span_choice"`.

**N-best rescoring** (`config.llm_mode = "nbest_rescore"`): Scores every full hypothesis with the LLM (conditioned on retrieved evidence), combines with acoustic + old LM scores using `alpha * llm + (1-alpha) * old_lm + acoustic_scale * acoustic`. Traces tag with `mode: "nbest_rescore"`.

Every sentence also gets a top-level `decision_mode` field: `"kept_top1"` (no uncertain spans, no LLM invoked), `"span_choice"`, or `"nbest_rescore"`.

## Key Interfaces

### Input: N-best from speechBCI

Produced by `lmDecoderUtils.nbest_with_lm_decoder()` at `NeuralDecoder/neuralDecoder/utils/lmDecoderUtils.py:207-229`. Returns `List[List[(sentence: str, ac_score: float, lm_score: float)]]` -- one list of N-best hypotheses per utterance.

### Input: Inference output dict

Produced by `NeuralSequenceDecoder.inference()`. Dict with keys: `logits` (numpy), `logitLengths` (numpy), `transcriptions` (numpy, null-terminated byte arrays), `trueSeqs`, `decodedSeqs`, `editDistances`, `trueSeqLengths`.

### Output: Pipeline result

`decode_with_confusion_rag()` returns:
```python
{
    "decoded_transcripts": List[str],
    "confidences": List[float],
    "trace_path": str | None,
    "trace": RunTrace,
}
```

### Output: Evaluation report

`full_evaluation()` returns a dict with: `wer`, `wer_ci_95`, `cer`, `top1_wer`, `oracle_wer`, `oracle_gap_closed`, `high_uncertainty_wer`, `rare_word_wer`, `faithfulness` (per-mode accuracy), `mode_breakdown_wer`, `summary`.

## Configuration

All hyperparameters live in `ConfusionRAGConfig` (see `confusionrag/config.py`). Key parameters to tune:

| Parameter | Default | Effect |
|---|---|---|
| `gate_metric` | `"entropy"` | Which uncertainty metric triggers retrieval |
| `gate_threshold` | `0.5` | Higher = fewer spans flagged uncertain = less retrieval |
| `llm_mode` | `"nbest_rescore"` | `"span_choice"` for per-word, `"nbest_rescore"` for full-hypothesis |
| `llm_alpha` | `0.5` | Weight for LLM score vs. old LM score (only in nbest_rescore) |
| `acoustic_scale` | `0.5` | Weight for acoustic score (only in nbest_rescore) |
| `retrieval_top_k` | `5` | Number of BM25 docs retrieved per uncertain span |
| `trace_enabled` | `True` | Set False to skip trace file I/O |

## Running Benchmarks

Benchmarks expect pre-computed N-best outputs and inference outputs saved as `.npy` files. These come from running the upstream speechBCI notebooks (`rnn_step3_baselineRNNInference.ipynb`, `5gram+llm_rescoring.ipynb`).

```bash
# Run baselines
python benchmarks/run_baselines.py \
    --nbest_path /path/to/nbest.npy \
    --inf_path /path/to/inference_out.npy \
    --corpus_path /path/to/train_transcriptions.txt \
    --output_dir ./results

# Run confusion-set RAG
python benchmarks/run_confusion_rag.py \
    --nbest_path /path/to/nbest.npy \
    --inf_path /path/to/inference_out.npy \
    --corpus_path /path/to/train_transcriptions.txt \
    --output_dir ./results \
    --llm_mode nbest_rescore \
    --gate_threshold 0.5

# Compare
python benchmarks/compare_results.py --results_dir ./results
```

## Tracing and Observability

Every pipeline run writes a JSON trace to `trace_dir` (default `./traces/`). The trace hierarchy is:

```
RunTrace
├── config (full snapshot)
├── summary (aggregate counts by mode)
└── sentences[]
    ├── decision_mode: "kept_top1" | "span_choice" | "nbest_rescore"
    └── spans[]
        ├── gate_result: "uncertain" | "confident"
        ├── retrieval: {query, docs} | null
        └── llm_decision: {mode, candidate_scores, selected, changed_from_top1} | null
```

Load traces with `RunTrace.load(path)`. The `RunSummary` gives at-a-glance mode distribution without parsing individual sentences.

## Gotchas

1. **Empty corpus**: `Retriever([])` is safe -- it sets `self._bm25 = None` and returns empty results. But an empty list passed directly to `BM25Okapi()` causes `ZeroDivisionError`.

2. **Trace `_start_time`**: `SentenceTrace` has a private `_start_time` dataclass field used by `start_timer()`/`stop_timer()`. Both `to_dict()` and `RunTrace.load()` explicitly strip it. If you add new private fields, follow the same pattern.

3. **Ground truth extraction**: `_extract_transcriptions()` in `pipeline.py` reads null-terminated byte arrays from `inference_out["transcriptions"]` and converts via `chr()`. This mirrors the same logic in speechBCI's `lmDecoderUtils.py:348-356`. If the upstream format changes, both must be updated.

4. **Combined scoring formula**: N-best rescore uses `alpha * new_llm + (1-alpha) * old_lm + acoustic_scale * acoustic`. This mirrors speechBCI's `gpt2_lm_decode()` at `lmDecoderUtils.py:89` but with evidence conditioning on the LLM input.

5. **Test isolation**: Tests never require a real LLM, real neural data, or the speechBCI package to be installed. All LLM calls are mocked, and synthetic N-best lists are generated in-test.

## Detailed File Documentation

See `speechBCI/ConfusionRAG/CHANGELOG.md` for per-file line counts, function signatures, parameter documentation, and the complete list of all 101 tests with descriptions.
