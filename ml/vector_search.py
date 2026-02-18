"""
Vector search using FAISS and sentence transformers.
Enables semantic search over patient history and medical knowledge.
"""

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorSearch:
    """
    FAISS-based vector search for semantic similarity.
    Uses sentence transformers for embedding generation.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "data/vector_index.faiss",
        metadata_path: str = "data/vector_metadata.pkl",
    ):
        """
        Initialize vector search.

        Args:
            model_name: Sentence transformer model to use
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load document metadata
        """
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path

        # Initialize sentence transformer
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index: faiss.IndexFlatL2 | None = None
        self.metadata: list[dict[str, Any]] = []

        # Create data directory
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        # Try to load existing index
        self._try_load_index()

    def _try_load_index(self):
        """Try to load existing index from disk"""
        if Path(self.index_path).exists() and Path(self.metadata_path).exists():
            try:
                self.load()
            except Exception as e:
                print(f"Failed to load index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create new empty FAISS index"""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            Embedding matrix (n_texts, embedding_dim)
        """
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        return embeddings.astype("float32")

    def add_documents(self, documents: list[str], metadata: list[dict[str, Any]] | None = None):
        """
        Add documents to the index.

        Args:
            documents: List of text documents
            metadata: Optional metadata for each document
        """
        if not documents:
            return

        # Generate embeddings
        embeddings = self.embed_texts(documents)

        # Add to index
        if self.index is None:
            self._create_new_index()

        self.index.add(embeddings)

        # Store metadata
        if metadata is None:
            metadata = [{"text": doc, "id": i} for i, doc in enumerate(documents)]

        self.metadata.extend(metadata)

    def search(
        self, query: str, k: int = 5, return_distances: bool = False
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            k: Number of results to return
            return_distances: Whether to include distance scores

        Returns:
            List of matching documents with metadata
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Generate query embedding
        query_embedding = self.embed_texts([query])

        # Search index
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

        # Collect results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                if return_distances:
                    result["distance"] = float(distances[0][i])
                    # Convert L2 distance to similarity score (0-1)
                    result["similarity"] = 1.0 / (1.0 + float(distances[0][i]))
                results.append(result)

        return results

    def search_symptoms(self, symptoms: list[str], k: int = 5) -> list[dict[str, Any]]:
        """
        Search for similar symptom patterns.

        Args:
            symptoms: List of symptoms
            k: Number of results

        Returns:
            Similar cases from history
        """
        query = " ".join(symptoms)
        return self.search(query, k=k, return_distances=True)

    def get_context_for_query(self, symptoms: list[str], max_results: int = 3) -> str:
        """
        Get relevant context from history for a query.

        Args:
            symptoms: List of symptoms
            max_results: Maximum number of context items

        Returns:
            Formatted context string
        """
        results = self.search_symptoms(symptoms, k=max_results)

        if not results:
            return "No relevant history found."

        context_parts = []
        for i, result in enumerate(results, 1):
            text = result.get("text", "")
            similarity = result.get("similarity", 0)
            context_parts.append(f"{i}. {text} (similarity: {similarity:.2f})")

        return "\n".join(context_parts)

    def save(self):
        """Save index and metadata to disk"""
        if self.index is None or self.index.ntotal == 0:
            print("Nothing to save - index is empty")
            return

        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)

            # Save metadata
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)

            print(f"Saved index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error saving index: {e}")

    def load(self):
        """Load index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)

            # Load metadata
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

            print(f"Loaded index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error loading index: {e}")
            raise

    def clear(self):
        """Clear index and metadata"""
        self._create_new_index()

    def get_stats(self) -> dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "n_vectors": self.index.ntotal if self.index else 0,
            "n_metadata": len(self.metadata),
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
        }


def index_patient_history(vector_search: VectorSearch, history_records: list[dict[str, Any]]):
    """
    Index patient history records for semantic search.

    Args:
        vector_search: VectorSearch instance
        history_records: List of history records with symptoms, diagnosis, etc.
    """
    if not history_records:
        return

    documents = []
    metadata = []

    for record in history_records:
        # Create searchable text from record
        symptoms = record.get("symptoms", [])
        diagnosis = record.get("diagnosis", "")
        notes = record.get("notes", "")

        text_parts = []
        if symptoms:
            text_parts.append(f"Symptoms: {', '.join(symptoms)}")
        if diagnosis:
            text_parts.append(f"Diagnosis: {diagnosis}")
        if notes:
            text_parts.append(f"Notes: {notes}")

        if text_parts:
            document_text = " | ".join(text_parts)
            documents.append(document_text)
            metadata.append(
                {
                    "text": document_text,
                    "timestamp": record.get("timestamp"),
                    "symptoms": symptoms,
                    "diagnosis": diagnosis,
                    "record_id": record.get("id"),
                }
            )

    if documents:
        vector_search.add_documents(documents, metadata)
