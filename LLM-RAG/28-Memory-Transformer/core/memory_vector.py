"""
Core Memory Vector Module
"""
import numpy as np
from datetime import datetime

class MemoryVector:
    """Memory vector wrapper class"""

    def __init__(self, content: str, importance: float = 0.5):
        self.content = content
        self.importance = importance
        self.timestamp = datetime.now()
        self.access_count = 0
        self.embedding = self._generate_embedding()

    def _generate_embedding(self) -> np.ndarray:
        """Generate an embedding vector"""
        np.random.seed(hash(self.content) % 2**32)
        vector = np.random.randn(128).astype("float32")

        # Keyword-based enhancement
        keywords = ["exclusivity", "clause", "breach", "important", "contract"]
        for i, keyword in enumerate(keywords):
            if keyword in self.content:
                vector[i * 20 : (i + 1) * 20] += 0.5

        return vector / np.linalg.norm(vector)

    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
