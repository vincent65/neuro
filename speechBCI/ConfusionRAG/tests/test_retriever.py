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
