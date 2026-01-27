"""
Memory Transformer Main Model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, List
from core.memory_vector import MemoryVector
from core.memory_storage import MemoryStorage
from core.memory_retriever import MemoryRetriever

class MemoryTransformer:
    """FAISS-optimized Memory Transformer main model"""

    def __init__(self, short_limit: int = 15, long_limit: int = 100):
        self.storage = MemoryStorage(short_limit, long_limit)
        self.retriever = MemoryRetriever()

    def process(self, content: str) -> Dict:
        """Process input content"""
        # Compute importance and store memory
        importance = self._calculate_importance(content)
        memory = MemoryVector(content, importance)
        self.storage.add_memory(memory)

        return {
            "memory_status": self.storage.get_status(),
            "content": content
        }

    def _calculate_importance(self, content: str) -> float:
        """Calculate content importance"""
        score = 0.3

        # Keyword detection
        keywords = ["exclusivity", "liquidated damages", "important", "termination", "obligation"]
        for keyword in keywords:
            if keyword in content:
                score += 0.15

        # Length bonus
        if len(content) > 30:
            score += 0.1

        return min(1.0, score)

    def query(self, query: str, top_k: int = 5) -> List[MemoryVector]:
        """Query relevant memories"""
        return self.retriever.retrieve(query, self.storage, top_k)
