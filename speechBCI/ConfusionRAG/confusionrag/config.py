from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ConfusionRAGConfig:
    """Configuration for the confusion-set guided RAG decoding pipeline."""

    # --- Confusion set construction ---
    # Minimum number of N-best hypotheses required to build confusion sets
    min_nbest: int = 2

    # --- Uncertainty gating ---
    # Metric used for gating: "entropy", "margin", or "disagreement_mass"
    gate_metric: str = "entropy"
    # Threshold above which a span is considered uncertain (for entropy/disagreement_mass)
    # or below which it is uncertain (for margin). Set to 0.0 to disable gating (always retrieve).
    gate_threshold: float = 0.5

    # --- Retrieval ---
    retrieval_top_k: int = 5
    # Number of context words on each side of an uncertain span used to build the query
    retrieval_context_window: int = 5
    # Maximum number of prior decoded sentences kept in session memory
    session_memory_size: int = 10
    # Keep only the top weighted confusion candidates in OR-expanded queries
    # (<=0 means keep all).
    retrieval_max_query_candidates: int = 6
    # Drop low-probability confusion candidates from OR-expanded queries.
    retrieval_min_candidate_weight: float = 0.02
    # Enable retrieval score quality gate; low-quality retrieval falls back
    # to no-evidence rescoring.
    retrieval_quality_gate_enabled: bool = False
    # Minimum top retrieval score required to keep retrieval evidence.
    retrieval_quality_min_top_score: float = 0.0
    # Minimum top-vs-second retrieval score gap required to keep evidence.
    retrieval_quality_min_score_gap: float = 0.0
    # Minimum number of non-zero-scored docs needed to keep evidence.
    retrieval_quality_min_nonzero_docs: int = 0
    # Optional BM25 + semantic rerank hybrid retrieval.
    retrieval_semantic_rerank_enabled: bool = False
    retrieval_semantic_rerank_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_semantic_rerank_top_n: int = 20
    # Prototype retrieval channels.
    confusion_memory_enabled: bool = False
    confusion_memory_top_k: int = 0
    confusion_memory_window: int = 2
    phonetic_retrieval_enabled: bool = False
    phonetic_retrieval_top_k: int = 0

    # --- LLM decoding ---
    # Mode: "span_choice" or "nbest_rescore"
    llm_mode: str = "nbest_rescore"
    # Interpolation weight for the new LLM score vs. the old LM score (0-1)
    llm_alpha: float = 0.5
    # Acoustic score scaling factor
    acoustic_scale: float = 0.5
    # Minimum combined-score margin required to switch away from top-1 in
    # n-best rescoring mode. Set to 0.0 to allow any positive improvement.
    nbest_change_margin_threshold: float = 0.0
    # Maximum number of evidence documents injected into the LLM prompt.
    evidence_max_docs: int = 5
    # Length penalty applied to LLM log-probabilities
    llm_length_penalty: float = 0.0
    # Max hypotheses scored per LLM forward pass (<=0 means all-at-once).
    llm_batch_size: int = 0

    # --- N-gram decoder ---
    blank_penalty: float = 1.9459  # np.log(7) ≈ 1.9459

    # --- Tracing ---
    trace_enabled: bool = True
    trace_dir: str = "./traces"

    def to_dict(self) -> dict:
        return asdict(self)
