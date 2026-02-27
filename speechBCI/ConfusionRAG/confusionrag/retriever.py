"""
BM25-based retrieval module for confusion-set guided decoding.

Provides:
- Index construction over training-side sentences (no test leakage).
- OR-expansion query construction from confusion sets.
- A session-memory buffer of previously decoded sentences for topic continuity.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

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
        max_query_candidates: int = 0,
        min_candidate_weight: float = 0.0,
        semantic_rerank_enabled: bool = False,
        semantic_rerank_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        semantic_rerank_top_n: int = 20,
        confusion_memory_enabled: bool = False,
        confusion_memory_top_k: int = 0,
        confusion_memory_window: int = 2,
        phonetic_retrieval_enabled: bool = False,
        phonetic_retrieval_top_k: int = 0,
    ):
        from rank_bm25 import BM25Okapi

        self.corpus = corpus
        self.top_k = top_k
        self.context_window = context_window
        self.max_query_candidates = max_query_candidates
        self.min_candidate_weight = min_candidate_weight
        self.semantic_rerank_enabled = semantic_rerank_enabled
        self.semantic_rerank_model = semantic_rerank_model
        self.semantic_rerank_top_n = max(int(semantic_rerank_top_n), self.top_k)
        self.confusion_memory_enabled = confusion_memory_enabled
        self.confusion_memory_top_k = max(0, confusion_memory_top_k)
        self.confusion_memory_window = max(1, confusion_memory_window)
        self.phonetic_retrieval_enabled = phonetic_retrieval_enabled
        self.phonetic_retrieval_top_k = max(0, phonetic_retrieval_top_k)
        self._session_memory: deque[str] = deque(maxlen=session_memory_size)
        self._semantic_model = None

        if corpus:
            tokenized = [doc.lower().split() for doc in corpus]
            self._tokenized_corpus = tokenized
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._tokenized_corpus = []
            self._bm25 = None

        self._confusion_memory_index: Dict[str, List[int]] = {}
        if self.confusion_memory_enabled and self.confusion_memory_top_k > 0:
            self._build_confusion_memory_index()

        self._phonetic_index: Dict[str, List[int]] = {}
        if self.phonetic_retrieval_enabled and self.phonetic_retrieval_top_k > 0:
            self._build_phonetic_index()

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

        query_candidates = self._prune_query_candidates(span)
        if len(query_candidates) > 1:
            or_clause = "(" + " OR ".join(query_candidates) + ")"
        else:
            or_clause = query_candidates[0] if query_candidates else ""

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
            docs: List[str] = []
            if include_session_memory and self.top_k > 0:
                docs = list(self._session_memory)[-self.top_k :]
            return RetrievalResult(
                query=query,
                retrieved_docs=docs,
                scores=[0.0] * len(docs),
                retrieval_time_ms=elapsed,
            )

        # Strip OR-expansion syntax for BM25 tokenisation
        clean_query = query.replace("(", "").replace(")", "").replace(" OR ", " ")
        tokens = clean_query.lower().split()

        bm25_scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(bm25_scores)[::-1][: self.semantic_rerank_top_n]
        docs = [self.corpus[i] for i in top_indices]
        doc_scores = [float(bm25_scores[i]) for i in top_indices]

        if self.semantic_rerank_enabled and docs:
            docs, doc_scores = self._semantic_rerank(query, docs, doc_scores)
        else:
            docs = docs[: self.top_k]
            doc_scores = doc_scores[: self.top_k]

        docs, doc_scores = self._compact_doc_pairs(
            docs,
            doc_scores,
            max_docs=self.top_k,
            drop_non_positive=True,
        )

        if include_session_memory:
            slots = max(self.top_k - len(docs), 0)
            query_tokens = set(tokens)
            for mem in reversed(self._session_memory):
                if slots <= 0:
                    break
                if mem in docs:
                    continue
                # Only backfill session memory when retrieval already has room.
                # If we already have corpus hits, require lexical overlap.
                if docs and query_tokens:
                    mem_tokens = set(mem.lower().split())
                    if not query_tokens.intersection(mem_tokens):
                        continue
                docs.append(mem)
                doc_scores.append(0.0)
                slots -= 1

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
        t0 = time.perf_counter()
        query = self.build_query(sentence_words, span)
        base = self.retrieve(query)
        docs = list(base.retrieved_docs)
        scores = list(base.scores)

        if self.confusion_memory_enabled and self.confusion_memory_top_k > 0:
            for doc, score in self._retrieve_from_confusion_memory(sentence_words, span):
                if doc not in docs:
                    docs.append(doc)
                    scores.append(float(score))

        if self.phonetic_retrieval_enabled and self.phonetic_retrieval_top_k > 0:
            for doc, score in self._retrieve_from_phonetics(span):
                if doc not in docs:
                    docs.append(doc)
                    scores.append(float(score))

        max_docs = (
            max(0, self.top_k)
            + max(0, self.confusion_memory_top_k)
            + max(0, self.phonetic_retrieval_top_k)
        )
        docs, scores = self._compact_doc_pairs(
            docs,
            scores,
            max_docs=max_docs,
            drop_non_positive=True,
        )

        return RetrievalResult(
            query=query,
            retrieved_docs=docs,
            scores=scores,
            retrieval_time_ms=(time.perf_counter() - t0) * 1000,
        )

    # ------------------------------------------------------------------
    # Query candidate pruning
    # ------------------------------------------------------------------

    def _prune_query_candidates(self, span: ConfusionSpan) -> List[str]:
        candidates = list(span.candidates)
        if not candidates:
            return []

        raw_weights = np.array(span.weights, dtype=float)
        if len(raw_weights) != len(candidates):
            raw_weights = np.ones(len(candidates), dtype=float)

        pairs = list(zip(candidates, raw_weights.tolist()))
        pairs.sort(key=lambda x: x[1], reverse=True)

        if self.min_candidate_weight > 0:
            filtered = [p for p in pairs if p[1] >= self.min_candidate_weight]
        else:
            filtered = pairs

        if not filtered:
            filtered = pairs[:1]

        if self.max_query_candidates > 0:
            filtered = filtered[: self.max_query_candidates]

        # De-duplicate while preserving order after pruning.
        seen: set[str] = set()
        out: List[str] = []
        for word, _ in filtered:
            if word not in seen:
                out.append(word)
                seen.add(word)
        return out

    # ------------------------------------------------------------------
    # Hybrid BM25 + semantic rerank
    # ------------------------------------------------------------------

    def _semantic_rerank(
        self,
        query: str,
        docs: List[str],
        bm25_scores: List[float],
    ) -> Tuple[List[str], List[float]]:
        self._ensure_semantic_model()
        if self._semantic_model is None or not docs:
            return docs[: self.top_k], bm25_scores[: self.top_k]

        q_emb = self._semantic_model.encode([query], normalize_embeddings=True)[0]
        d_emb = self._semantic_model.encode(docs, normalize_embeddings=True)
        sims = np.dot(d_emb, q_emb)

        order = np.argsort(sims)[::-1][: self.top_k]
        reranked_docs = [docs[i] for i in order]
        reranked_scores = [float(sims[i]) for i in order]
        return reranked_docs, reranked_scores

    def _ensure_semantic_model(self) -> None:
        if self._semantic_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Semantic reranking requested, but sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            ) from exc
        self._semantic_model = SentenceTransformer(self.semantic_rerank_model)

    # ------------------------------------------------------------------
    # Prototype retrieval channels
    # ------------------------------------------------------------------

    def _build_confusion_memory_index(self) -> None:
        for doc_idx, words in enumerate(self._tokenized_corpus):
            for pos in range(len(words)):
                key = self._context_key(words, pos, pos + 1, self.confusion_memory_window)
                if key not in self._confusion_memory_index:
                    self._confusion_memory_index[key] = []
                if doc_idx not in self._confusion_memory_index[key]:
                    self._confusion_memory_index[key].append(doc_idx)

    def _retrieve_from_confusion_memory(
        self,
        sentence_words: List[str],
        span: ConfusionSpan,
    ) -> List[Tuple[str, float]]:
        if not self._confusion_memory_index:
            return []

        words = [w.lower() for w in sentence_words]
        key = self._context_key(words, span.span_start, span.span_end, self.confusion_memory_window)
        doc_ids = self._confusion_memory_index.get(key, [])
        if not doc_ids:
            return []

        cand_set = {c.lower() for c in span.candidates}
        ranked: List[Tuple[str, float]] = []
        for doc_idx in doc_ids:
            doc_words = self._tokenized_corpus[doc_idx]
            overlap = len(cand_set.intersection(doc_words))
            score = 0.2 + 0.05 * overlap
            ranked.append((self.corpus[doc_idx], float(score)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[: self.confusion_memory_top_k]

    def _build_phonetic_index(self) -> None:
        for doc_idx, words in enumerate(self._tokenized_corpus):
            seen_codes: set[str] = set()
            for word in words:
                code = self._soundex(word)
                if not code or code in seen_codes:
                    continue
                seen_codes.add(code)
                if code not in self._phonetic_index:
                    self._phonetic_index[code] = []
                self._phonetic_index[code].append(doc_idx)

    def _retrieve_from_phonetics(self, span: ConfusionSpan) -> List[Tuple[str, float]]:
        if not self._phonetic_index:
            return []

        doc_votes: Dict[int, int] = {}
        for cand in span.candidates:
            code = self._soundex(cand)
            if not code:
                continue
            for doc_idx in self._phonetic_index.get(code, []):
                doc_votes[doc_idx] = doc_votes.get(doc_idx, 0) + 1

        ranked = sorted(doc_votes.items(), key=lambda x: x[1], reverse=True)
        out: List[Tuple[str, float]] = []
        for doc_idx, votes in ranked[: self.phonetic_retrieval_top_k]:
            out.append((self.corpus[doc_idx], float(0.15 + 0.05 * votes)))
        return out

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    @staticmethod
    def _compact_doc_pairs(
        docs: List[str],
        scores: List[float],
        max_docs: int,
        drop_non_positive: bool = True,
    ) -> Tuple[List[str], List[float]]:
        if not docs or max_docs == 0:
            return [], []

        doc_to_score: Dict[str, float] = {}
        for idx, doc in enumerate(docs):
            score = float(scores[idx]) if idx < len(scores) else 0.0
            if doc not in doc_to_score or score > doc_to_score[doc]:
                doc_to_score[doc] = score

        pairs = sorted(doc_to_score.items(), key=lambda x: x[1], reverse=True)
        if drop_non_positive:
            positive_pairs = [p for p in pairs if p[1] > 0.0]
            pairs = positive_pairs if positive_pairs else []

        if max_docs > 0:
            pairs = pairs[:max_docs]

        compact_docs = [doc for doc, _ in pairs]
        compact_scores = [float(score) for _, score in pairs]
        return compact_docs, compact_scores

    @staticmethod
    def _context_key(words: List[str], start: int, end: int, window: int) -> str:
        left = words[max(0, start - window): start]
        right = words[end: min(len(words), end + window)]
        return " ".join(left) + " || " + " ".join(right)

    @staticmethod
    def _soundex(word: str) -> str:
        w = re.sub(r"[^a-z]", "", word.lower())
        if not w:
            return ""
        first = w[0].upper()
        mapping = {
            "bfpv": "1",
            "cgjkqsxz": "2",
            "dt": "3",
            "l": "4",
            "mn": "5",
            "r": "6",
        }
        code_map = {}
        for chars, value in mapping.items():
            for ch in chars:
                code_map[ch] = value

        encoded = []
        prev = ""
        for ch in w[1:]:
            digit = code_map.get(ch, "0")
            if digit != prev:
                encoded.append(digit)
            prev = digit

        encoded = [d for d in encoded if d != "0"]
        return (first + "".join(encoded) + "000")[:4]
