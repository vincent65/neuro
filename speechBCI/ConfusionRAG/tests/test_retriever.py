"""Tests for retriever.py: BM25 index, query construction, session memory."""

import pytest

from confusionrag.confusion_set import ConfusionSpan
from confusionrag.retriever import Retriever, RetrievalResult
import numpy as np


@pytest.fixture
def small_corpus():
    return [
        "the green library is a great place to study",
        "stanford has many libraries on campus",
        "the engineering library is near the quad",
        "machine learning is transforming neuroscience",
        "brain computer interfaces decode neural signals",
    ]


@pytest.fixture
def retriever(small_corpus):
    return Retriever(small_corpus, top_k=3, context_window=2, session_memory_size=5)


class TestRetrieverInit:
    def test_creates_index(self, retriever):
        assert retriever.corpus is not None
        assert len(retriever.corpus) == 5

    def test_empty_corpus(self):
        r = Retriever([], top_k=3)
        result = r.retrieve("test query")
        assert isinstance(result, RetrievalResult)


class TestQueryConstruction:
    def test_or_expansion(self, retriever):
        span = ConfusionSpan(
            span_start=1, span_end=2,
            candidates=["green", "greene", "cream"],
            weights=np.array([0.5, 0.3, 0.2]),
        )
        query = retriever.build_query(
            ["the", "green", "library", "is", "open"], span
        )
        assert "OR" in query
        assert "green" in query
        assert "greene" in query
        assert "cream" in query

    def test_single_candidate_no_or(self, retriever):
        span = ConfusionSpan(
            span_start=1, span_end=2,
            candidates=["green"],
            weights=np.array([1.0]),
        )
        query = retriever.build_query(
            ["the", "green", "library"], span
        )
        assert "OR" not in query
        assert "green" in query

    def test_context_window(self, retriever):
        words = ["a", "b", "c", "d", "e", "f", "g"]
        span = ConfusionSpan(
            span_start=3, span_end=4,
            candidates=["d", "x"],
            weights=np.array([0.6, 0.4]),
        )
        query = retriever.build_query(words, span)
        assert "b" in query or "c" in query
        assert "e" in query or "f" in query

    def test_query_candidate_pruning_max_candidates(self, small_corpus):
        r = Retriever(
            small_corpus,
            top_k=3,
            max_query_candidates=2,
        )
        span = ConfusionSpan(
            span_start=1, span_end=2,
            candidates=["green", "greene", "cream"],
            weights=np.array([0.7, 0.2, 0.1]),
        )
        query = r.build_query(["the", "green", "library"], span)
        assert "green" in query
        assert "greene" in query
        assert "cream" not in query

    def test_query_candidate_pruning_min_weight(self, small_corpus):
        r = Retriever(
            small_corpus,
            top_k=3,
            min_candidate_weight=0.25,
        )
        span = ConfusionSpan(
            span_start=1, span_end=2,
            candidates=["green", "greene", "cream"],
            weights=np.array([0.7, 0.2, 0.1]),
        )
        query = r.build_query(["the", "green", "library"], span)
        assert "green" in query
        assert "greene" not in query
        assert "cream" not in query


class TestRetrieval:
    def test_returns_retrieval_result(self, retriever):
        result = retriever.retrieve("green library")
        assert isinstance(result, RetrievalResult)
        assert len(result.retrieved_docs) > 0
        assert result.retrieval_time_ms >= 0

    def test_top_k_limit(self, retriever):
        result = retriever.retrieve("library")
        corpus_docs = [d for d in result.retrieved_docs if d in retriever.corpus]
        assert len(corpus_docs) <= 3

    def test_relevant_doc_ranked_high(self, retriever):
        result = retriever.retrieve("green library study")
        assert "the green library is a great place to study" in result.retrieved_docs[:2]

    def test_retrieve_for_span(self, retriever):
        span = ConfusionSpan(
            span_start=1, span_end=2,
            candidates=["green", "engineering"],
            weights=np.array([0.6, 0.4]),
        )
        result = retriever.retrieve_for_span(
            ["the", "green", "library"], span
        )
        assert isinstance(result, RetrievalResult)
        assert result.query != ""

    def test_zero_score_docs_are_filtered(self, retriever):
        result = retriever.retrieve("qwertyuiop asdfghjkl")
        assert result.retrieved_docs == []
        assert result.scores == []

    def test_semantic_rerank_path_can_be_overridden(self, small_corpus):
        base = Retriever(small_corpus, top_k=2)
        baseline_docs = base.retrieve("library").retrieved_docs[:2]
        assert len(baseline_docs) == 2

        reranked = Retriever(
            small_corpus,
            top_k=2,
            semantic_rerank_enabled=True,
            semantic_rerank_top_n=4,
        )
        reranked._semantic_rerank = lambda q, docs, scores: (list(reversed(docs[:2])), [1.0, 0.9])
        result = reranked.retrieve("library")
        assert result.retrieved_docs[:2] == list(reversed(baseline_docs))

    def test_confusion_memory_channel_adds_docs(self):
        corpus = [
            "the green library is open",
            "the engineering library is near campus",
            "machine learning systems are useful",
        ]
        r = Retriever(
            corpus,
            top_k=1,
            context_window=1,
            confusion_memory_enabled=True,
            confusion_memory_top_k=2,
            confusion_memory_window=1,
        )
        span = ConfusionSpan(
            span_start=1, span_end=2,
            candidates=["green", "engineering"],
            weights=np.array([0.6, 0.4]),
        )
        result = r.retrieve_for_span(
            ["the", "green", "library", "is", "open"],
            span,
        )
        assert "the engineering library is near campus" in result.retrieved_docs

    def test_phonetic_channel_can_retrieve_without_bm25_docs(self):
        corpus = [
            "two dogs run fast",
            "the tone is too high",
            "he drew a circle",
        ]
        r = Retriever(
            corpus,
            top_k=0,
            phonetic_retrieval_enabled=True,
            phonetic_retrieval_top_k=2,
        )
        span = ConfusionSpan(
            span_start=0, span_end=1,
            candidates=["to", "two", "too"],
            weights=np.array([0.4, 0.4, 0.2]),
        )
        result = r.retrieve_for_span(["to", "dogs", "run"], span)
        assert len(result.retrieved_docs) > 0


class TestSessionMemory:
    def test_add_and_get(self, retriever):
        retriever.add_to_session_memory("first decoded sentence")
        retriever.add_to_session_memory("second decoded sentence")
        mem = retriever.get_session_memory()
        assert len(mem) == 2
        assert "first decoded sentence" in mem

    def test_memory_limit(self):
        r = Retriever(["doc"], top_k=1, session_memory_size=2)
        r.add_to_session_memory("a")
        r.add_to_session_memory("b")
        r.add_to_session_memory("c")
        mem = r.get_session_memory()
        assert len(mem) == 2
        assert "a" not in mem

    def test_session_memory_in_retrieval(self, retriever):
        retriever.add_to_session_memory("a very unique session sentence xyz")
        result = retriever.retrieve("anything")
        assert "a very unique session sentence xyz" in result.retrieved_docs

    def test_session_memory_not_appended_when_bm25_hits_fill_budget(self, retriever):
        retriever.add_to_session_memory("a very unique session sentence xyz")
        result = retriever.retrieve("green library")
        assert "a very unique session sentence xyz" not in result.retrieved_docs

    def test_clear_memory(self, retriever):
        retriever.add_to_session_memory("something")
        retriever.clear_session_memory()
        assert len(retriever.get_session_memory()) == 0

    def test_to_dict(self, retriever):
        result = retriever.retrieve("test")
        d = result.to_dict()
        assert "query" in d
        assert "retrieved_docs" in d
        assert isinstance(d["scores"], list)
