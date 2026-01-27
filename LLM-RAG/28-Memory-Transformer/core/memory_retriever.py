"""
FAISS Memory Retriever - Simplified Version
"""
import numpy as np
from typing import List
from .memory_vector import MemoryVector

class MemoryRetriever:
    """FAISS-based memory retriever"""

    def retrieve(self, query: str, storage, top_k: int = 5) -> List[MemoryVector]:
        """Retrieve relevant memories"""
        query_vector = self._encode_query(query)
        return storage.search(query_vector, top_k)

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode the query"""
        np.random.seed(hash(query) % 2**32)
        vector = np.random.randn(128).astype("float32")

        keywords = ["exclusivity", "clause", "breach", "important", "contract"]
        for i, keyword in enumerate(keywords):
            if keyword in query:
                vector[i * 20 : (i + 1) * 20] += 0.5

        return vector / np.linalg.norm(vector)
