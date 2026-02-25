"""
BM25-based retrieval module for confusion-set guided decoding.

Provides:
- Index construction over training-side sentences (no test leakage).
- OR-expansion query construction from confusion sets.
- A session-memory buffer of previously decoded sentences for topic continuity.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np

from confusionrag.confusion_set import ConfusionSpan


# ---------------------------------------------------------------------------
# Retrieval result
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    query: str
    retrieved_docs: List[str]
    scores: List[float]
    retrieval_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieved_docs": self.retrieved_docs,
            "scores": [float(s) for s in self.scores],
            "retrieval_time_ms": self.retrieval_time_ms,
        }


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Lightweight BM25 retriever backed by ``rank_bm25``.

    Parameters
    ----------
    corpus : list of str
        Training sentences to index.  Each string is one document.
    top_k : int
        Number of documents to return per query.
    context_window : int
        Number of words on each side of a confusion span used when building
        the retrieval query.
    session_memory_size : int
        Maximum number of recently decoded sentences stored for session
        continuity retrieval.
    """

    def __init__(
        self,
        corpus: List[str],
        top_k: int = 5,
        context_window: int = 5,
        session_memory_size: int = 10,
    ):
        from rank_bm25 import BM25Okapi

        self.corpus = corpus
        self.top_k = top_k
        self.context_window = context_window
        self._session_memory: deque[str] = deque(maxlen=session_memory_size)

        if corpus:
            tokenized = [doc.lower().split() for doc in corpus]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    # ------------------------------------------------------------------
    # Session memory
    # ------------------------------------------------------------------

    def add_to_session_memory(self, sentence: str) -> None:
        self._session_memory.append(sentence)

    def get_session_memory(self) -> List[str]:
        return list(self._session_memory)

    def clear_session_memory(self) -> None:
        self._session_memory.clear()

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------

    def build_query(
        self,
        sentence_words: List[str],
        span: ConfusionSpan,
    ) -> str:
        """
        Build a retrieval query from the sentence context around an uncertain
        span, expanding the span position with an OR over confusion candidates.

        Example: ``"... sat at (green OR greene OR cream) library ..."``
        """
        start = max(0, span.span_start - self.context_window)
        end = min(len(sentence_words), span.span_end + self.context_window)

        left_ctx = sentence_words[start : span.span_start]
        right_ctx = sentence_words[span.span_end : end]

        if len(span.candidates) > 1:
            or_clause = "(" + " OR ".join(span.candidates) + ")"
        else:
            or_clause = span.candidates[0] if span.candidates else ""

        parts = left_ctx + [or_clause] + right_ctx
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        include_session_memory: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve top-k documents from the corpus and (optionally) append
        recent session-memory sentences as additional context.
        """
        t0 = time.perf_counter()

        if self._bm25 is None or len(self.corpus) == 0:
            elapsed = (time.perf_counter() - t0) * 1000
            docs = list(self._session_memory) if include_session_memory else []
            return RetrievalResult(
                query=query,
                retrieved_docs=docs,
                scores=[0.0] * len(docs),
                retrieval_time_ms=elapsed,
            )

        # Strip OR-expansion syntax for BM25 tokenisation
        clean_query = query.replace("(", "").replace(")", "").replace(" OR ", " ")
        tokens = clean_query.lower().split()

        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][: self.top_k]

        docs = [self.corpus[i] for i in top_indices]
        doc_scores = [float(scores[i]) for i in top_indices]

        if include_session_memory:
            for mem in self._session_memory:
                if mem not in docs:
                    docs.append(mem)
                    doc_scores.append(0.0)

        elapsed = (time.perf_counter() - t0) * 1000
        return RetrievalResult(
            query=query,
            retrieved_docs=docs,
            scores=doc_scores,
            retrieval_time_ms=elapsed,
        )

    def retrieve_for_span(
        self,
        sentence_words: List[str],
        span: ConfusionSpan,
    ) -> RetrievalResult:
        """Convenience: build query then retrieve."""
        query = self.build_query(sentence_words, span)
        return self.retrieve(query)
