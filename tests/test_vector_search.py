"""
Tests for vector search module.
NOTE: Tests use mocked sentence transformer to avoid network dependencies.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ml.vector_search import VectorSearch, index_patient_history


@pytest.fixture
def mock_encoder():
    """Mock sentence transformer encoder"""
    mock = Mock()
    mock.get_sentence_embedding_dimension.return_value = 384
    
    def mock_encode(texts, convert_to_numpy=True):
        # Return embeddings with shape (len(texts), 384)
        return np.random.rand(len(texts) if isinstance(texts, list) else 1, 384).astype("float32")
    
    mock.encode = mock_encode
    return mock


@pytest.fixture
def temp_paths():
    """Create temporary paths for index and metadata"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index.faiss"
        metadata_path = Path(tmpdir) / "test_metadata.pkl"
        yield str(index_path), str(metadata_path)


@pytest.fixture
def vector_search(temp_paths, mock_encoder):
    """Create vector search instance for testing"""
    index_path, metadata_path = temp_paths
    
    with patch("ml.vector_search.SentenceTransformer") as mock_st:
        mock_st.return_value = mock_encoder
        vs = VectorSearch(index_path=index_path, metadata_path=metadata_path)
        yield vs


class TestVectorSearch:
    """Test suite for VectorSearch"""

    def test_initialization(self, temp_paths, mock_encoder):
        """Test vector search initialization"""
        index_path, metadata_path = temp_paths
        
        with patch("ml.vector_search.SentenceTransformer") as mock_st:
            mock_st.return_value = mock_encoder
            vs = VectorSearch(index_path=index_path, metadata_path=metadata_path)

            assert vs.encoder is not None
            assert vs.embedding_dim > 0
            assert vs.index is not None
            assert isinstance(vs.metadata, list)

    def test_add_and_search(self, vector_search):
        """Test adding documents and searching"""
        documents = [
            "Patient has headache and fever",
            "Experiencing chest pain",
            "Reports shortness of breath",
        ]

        vector_search.add_documents(documents)
        assert vector_search.index.ntotal == 3

        # Search
        results = vector_search.search("headache", k=2)
        assert len(results) == 2

    def test_search_empty_index(self, vector_search):
        """Test searching empty index"""
        results = vector_search.search("test", k=5)
        assert len(results) == 0

    def test_get_stats(self, vector_search):
        """Test getting index statistics"""
        stats = vector_search.get_stats()
        assert "n_vectors" in stats
        assert stats["n_vectors"] == 0

        # Add documents
        vector_search.add_documents(["doc1", "doc2"])
        stats = vector_search.get_stats()
        assert stats["n_vectors"] == 2

    def test_clear(self, vector_search):
        """Test clearing index"""
        vector_search.add_documents(["doc1", "doc2"])
        assert vector_search.index.ntotal == 2

        vector_search.clear()
        assert vector_search.index.ntotal == 0

    def test_save_and_load(self, temp_paths, mock_encoder):
        """Test saving and loading"""
        index_path, metadata_path = temp_paths

        with patch("ml.vector_search.SentenceTransformer") as mock_st:
            mock_st.return_value = mock_encoder
            
            # Create and save
            vs1 = VectorSearch(index_path=index_path, metadata_path=metadata_path)
            vs1.add_documents(["doc1", "doc2", "doc3"])
            vs1.save()

            assert Path(index_path).exists()

            # Load
            vs2 = VectorSearch(index_path=index_path, metadata_path=metadata_path)
            assert vs2.index.ntotal == 3


class TestIndexPatientHistory:
    """Test suite for patient history indexing"""

    def test_index_patient_history(self, vector_search):
        """Test indexing patient history"""
        history = [
            {
                "id": 1,
                "symptoms": ["fever", "cough"],
                "diagnosis": "Upper respiratory infection",
                "timestamp": "2024-01-01",
            },
        ]

        index_patient_history(vector_search, history)
        assert vector_search.index.ntotal == 1

    def test_index_empty_history(self, vector_search):
        """Test indexing empty history"""
        index_patient_history(vector_search, [])
        assert vector_search.index.ntotal == 0
