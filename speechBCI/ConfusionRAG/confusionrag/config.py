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

    # --- LLM decoding ---
    # Mode: "span_choice" or "nbest_rescore"
    llm_mode: str = "nbest_rescore"
    # Interpolation weight for the new LLM score vs. the old LM score (0-1)
    llm_alpha: float = 0.5
    # Acoustic score scaling factor
    acoustic_scale: float = 0.5
    # Length penalty applied to LLM log-probabilities
    llm_length_penalty: float = 0.0

    # --- N-gram decoder ---
    blank_penalty: float = 1.9459  # np.log(7) ≈ 1.9459

    # --- Tracing ---
    trace_enabled: bool = True
    trace_dir: str = "./traces"

    def to_dict(self) -> dict:
        return asdict(self)
