"""
test_mongo_vector_store.py — Unit tests for tools/mongo_vector_store.py

_cosine_similarity is a pure function tested without any mocks.
All MongoDB operations are tested by patching _get_collection.
"""

import pytest
from unittest.mock import MagicMock, patch

from tools.mongo_vector_store import (
    _cosine_similarity,
    store_document,
    clear_documents,
    count_documents,
    similarity_search,
)


# ── _cosine_similarity (pure math) ───────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_return_1(self):
        result = _cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert abs(result - 1.0) < 1e-9

    def test_opposite_vectors_return_minus_1(self):
        result = _cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(result - (-1.0)) < 1e-9

    def test_orthogonal_vectors_return_0(self):
        result = _cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(result) < 1e-9

    def test_zero_vector_returns_0(self):
        """Denominator is 0 — should return 0.0 without error."""
        result = _cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        assert result == 0.0

    def test_both_zero_vectors_return_0(self):
        result = _cosine_similarity([0.0, 0.0], [0.0, 0.0])
        assert result == 0.0

    def test_scaled_same_direction_returns_1(self):
        """Cosine similarity is invariant to vector magnitude."""
        result = _cosine_similarity([1.0, 0.0], [5.0, 0.0])
        assert abs(result - 1.0) < 1e-9

    def test_partial_overlap_returns_value_between_0_and_1(self):
        result = _cosine_similarity([1.0, 1.0], [1.0, 0.0])
        assert 0.0 < result < 1.0


# ── store_document ────────────────────────────────────────────────────────────

class TestStoreDocument:
    def test_insert_one_called_with_correct_fields(self):
        mock_col = MagicMock()
        mock_col.insert_one.return_value.inserted_id = "fake-id"
        with patch("tools.mongo_vector_store._get_collection", return_value=mock_col):
            store_document("some text", [0.1, 0.2], "source.pdf")
        call_doc = mock_col.insert_one.call_args[0][0]
        assert call_doc["text"] == "some text"
        assert call_doc["embedding"] == [0.1, 0.2]
        assert call_doc["source"] == "source.pdf"

    def test_returns_inserted_id_as_string(self):
        mock_col = MagicMock()
        mock_col.insert_one.return_value.inserted_id = "abc123"
        with patch("tools.mongo_vector_store._get_collection", return_value=mock_col):
            result = store_document("text", [0.0], "src")
        assert isinstance(result, str)
        assert result == "abc123"

    def test_default_source_is_manual(self):
        mock_col = MagicMock()
        mock_col.insert_one.return_value.inserted_id = "x"
        with patch("tools.mongo_vector_store._get_collection", return_value=mock_col):
            store_document("text", [0.0])
        call_doc = mock_col.insert_one.call_args[0][0]
        assert call_doc["source"] == "manual"


# ── clear_documents ───────────────────────────────────────────────────────────

class TestClearDocuments:
    def test_returns_deleted_count(self):
        mock_col = MagicMock()
        mock_col.delete_many.return_value.deleted_count = 7
        with patch("tools.mongo_vector_store._get_collection", return_value=mock_col):
            result = clear_documents()
        assert result == 7

    def test_calls_delete_many_with_empty_filter(self):
        mock_col = MagicMock()
        mock_col.delete_many.return_value.deleted_count = 0
        with patch("tools.mongo_vector_store._get_collection", return_value=mock_col):
            clear_documents()
        mock_col.delete_many.assert_called_once_with({})

    def test_returns_zero_when_collection_empty(self):
        mock_col = MagicMock()
        mock_col.delete_many.return_value.deleted_count = 0
        with patch("tools.mongo_vector_store._get_collection", return_value=mock_col):
            result = clear_documents()
        assert result == 0


# ── count_documents ───────────────────────────────────────────────────────────

class TestCountDocuments:
    def test_returns_count_from_mongo(self):
        mock_col = MagicMock()
        mock_col.count_documents.return_value = 42
        with patch("tools.mongo_vector_store._get_collection", return_value=mock_col):
            result = count_documents()
        assert result == 42

    def test_returns_zero_for_empty_collection(self):
        mock_col = MagicMock()
        mock_col.count_documents.return_value = 0
        with patch("tools.mongo_vector_store._get_collection", return_value=mock_col):
            result = count_documents()
        assert result == 0


# ── similarity_search ─────────────────────────────────────────────────────────

class TestSimilaritySearch:
    # Five test documents with known vectors
    _DOCS = [
        {"text": "doc1", "embedding": [1.0, 0.0], "source": "a.pdf"},   # closest to [1,0]
        {"text": "doc2", "embedding": [0.0, 1.0], "source": "b.pdf"},
        {"text": "doc3", "embedding": [0.7, 0.7], "source": "c.pdf"},
        {"text": "doc4", "embedding": [-1.0, 0.0], "source": "d.pdf"},  # farthest
        {"text": "doc5", "embedding": [0.5, 0.5], "source": "e.pdf"},
    ]

    def test_empty_collection_returns_empty_list(self):
        mock_col = MagicMock()
        mock_col.find.return_value = []
        with patch("tools.mongo_vector_store._get_collection", return_value=mock_col):
            result = similarity_search([1.0, 0.0], top_k=3)
        assert result == []

    def test_returns_top_k_results(self):
        mock_col = MagicMock()
        mock_col.find.return_value = self._DOCS
        with patch("tools.mongo_vector_store._get_collection", return_value=mock_col):
            result = similarity_search([1.0, 0.0], top_k=3)
        assert len(result) == 3

    def test_results_sorted_by_score_descending(self):
        mock_col = MagicMock()
        mock_col.find.return_value = self._DOCS
        with patch("tools.mongo_vector_store._get_collection", return_value=mock_col):
            result = similarity_search([1.0, 0.0], top_k=5)
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_best_match_is_identical_direction(self):
        mock_col = MagicMock()
        mock_col.find.return_value = self._DOCS
        with patch("tools.mongo_vector_store._get_collection", return_value=mock_col):
            result = similarity_search([1.0, 0.0], top_k=1)
        assert result[0][0] == "doc1"
        assert abs(result[0][1] - 1.0) < 1e-9

    def test_returns_correct_tuple_shape(self):
        mock_col = MagicMock()
        mock_col.find.return_value = [{"text": "t", "embedding": [1.0, 0.0], "source": "s"}]
        with patch("tools.mongo_vector_store._get_collection", return_value=mock_col):
            result = similarity_search([1.0, 0.0], top_k=1)
        assert len(result) == 1
        text, score, source = result[0]
        assert isinstance(text, str)
        assert isinstance(score, float)
        assert isinstance(source, str)

    def test_k_larger_than_docs_returns_all_docs(self):
        mock_col = MagicMock()
        mock_col.find.return_value = self._DOCS[:2]
        with patch("tools.mongo_vector_store._get_collection", return_value=mock_col):
            result = similarity_search([1.0, 0.0], top_k=10)
        assert len(result) == 2

    def test_source_field_included_in_result(self):
        mock_col = MagicMock()
        mock_col.find.return_value = [{"text": "t", "embedding": [1.0, 0.0], "source": "my_doc.pdf"}]
        with patch("tools.mongo_vector_store._get_collection", return_value=mock_col):
            result = similarity_search([1.0, 0.0], top_k=1)
        assert result[0][2] == "my_doc.pdf"
