import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

"""
Gumbel-Softmax Tool Selector Demo
--------------------------------
This example demonstrates the core use of Gumbel-Softmax for tool selection.
Gumbel-Softmax enables differentiable discrete choices during both
forward and backward passes.
"""

# =======================
# 1. Tool Definitions – three similar but distinct search tools
# =======================
def academic_search(query):
    """Academic search tool – used to retrieve research papers and scholarly materials"""
    return f"[Academic Search] Found academic papers related to '{query}'"

def news_search(query):
    """News search tool – used to retrieve latest news and current events"""
    return f"[News Search] Found latest news related to '{query}'"

def wiki_search(query):
    """Encyclopedia search tool – used to retrieve background knowledge and definitions"""
    return f"[Wiki Search] Found encyclopedia explanations related to '{query}'"

TOOLS = {
    "academic": academic_search,
    "news": news_search,
    "wiki": wiki_search
}

# =======================
# 2. Gumbel-Softmax Tool Selector – core component
# =======================
class ToolSelector(nn.Module):
    def __init__(self, num_tools=3, input_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_tools)
        )
        self.tool_names = list(TOOLS.keys())
    
    def forward(self, x, tau=0.5, hard=False):
        logits = self.model(x)
        
        # Gumbel-Softmax sampling – core logic
        # 1. Add Gumbel noise: logits + g, where g ~ Gumbel(0,1)
        # 2. Apply softmax: softmax((logits + g) / tau)
        # 3. tau is the temperature parameter:
        #    - high tau → more uniform (exploration)
        #    - low tau → sharper (exploitation)
        y_soft = nn.functional.gumbel_softmax(
            logits, tau=tau, hard=hard, dim=-1
        )
        
        return y_soft, logits
    
    def predict(self, x):
        with torch.no_grad():
            _, logits = self.forward(x)
            return torch.argmax(logits, dim=-1).item(), logits[0]

# =======================
# 3. FAISS-based Text Encoder
# =======================
class FAISSTextEncoder:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize FAISS-based text encoder
        
        Args:
            model_name: name of the pretrained SentenceTransformer model
        """
        # 1. Load pretrained embedding model
        print("Loading text embedding model...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384  # MiniLM embedding dimension
        
        # 2. Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # inner product similarity
        
        # 3. Build representative query knowledge base for tool categories
        self._build_tool_knowledge_base()
        print("FAISS text encoder initialized successfully")
    
    def _build_tool_knowledge_base(self):
        """Build representative query knowledge base for each tool category"""
        tool_queries = {
            "academic": [
                "deep learning research papers",
                "machine learning algorithm analysis",
                "artificial intelligence theoretical research",
                "data mining techniques survey",
                "neural network model experiments",
                "academic literature review",
                "latest progress in computer vision",
                "natural language processing research methods",
            ],
            "news": [
                "latest technology news",
                "today's AI company updates",
                "technology company funding news",
                "artificial intelligence industry reports",
                "technology breakthrough news",
                "market trend updates",
                "new product releases this week",
                "recently announced mergers and acquisitions",
            ],
            "wiki": [
                "basic concepts of artificial intelligence",
                "machine learning definition",
                "introduction to deep learning principles",
                "algorithm fundamentals",
                "technical terminology explanation",
                "historical development overview",
                "what is a neural network",
                "basic principles of reinforcement learning",
            ],
        }
        
        all_queries = []
        self.query_to_tool = {}
        
        for tool, queries in tool_queries.items():
            for query in queries:
                all_queries.append(query)
                self.query_to_tool[query] = tool
        
        print(f"Generating embeddings for {len(all_queries)} reference queries...")
        embeddings = self.model.encode(all_queries)
        self.index.add(embeddings.astype("float32"))
        self.reference_queries = all_queries
    
    def encode_query(self, query):
        """
        Convert a query string into a feature vector
        
        Args:
            query: input query text
            
        Returns:
            torch.Tensor: feature vector
        """
        query_embedding = self.model.encode([query])
        
        k = 6
        similarities, indices = self.index.search(
            query_embedding.astype("float32"), k
        )
        
        feature_vector = np.zeros(64)
        tool_counts = {"academic": 0, "news": 0, "wiki": 0}
        
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= len(self.reference_queries):
                continue
            
            ref_query = self.reference_queries[idx]
            tool = self.query_to_tool[ref_query]
            tool_idx = ["academic", "news", "wiki"].index(tool)
            base_idx = tool_idx * 20
            
            weight = max(0, sim)
            feature_idx = base_idx + tool_counts[tool]
            if feature_idx < base_idx + 15:
                feature_vector[feature_idx] = weight
                tool_counts[tool] += 1
        
        if any(w in query for w in ["today", "latest", "just"]):
            feature_vector[1 * 20 + 15] = 1.0
        if any(w in query for w in ["research", "paper"]):
            feature_vector[0 * 20 + 15] = 1.0
        if any(w in query for w in ["what is", "definition", "concept"]):
            feature_vector[2 * 20 + 15] = 1.0
        
        return torch.FloatTensor(feature_vector).unsqueeze(0)

# Global encoder instance
text_encoder = None

def encode_query(query):
    global text_encoder
    if text_encoder is None:
        text_encoder = FAISSTextEncoder()
    return text_encoder.encode_query(query)

# =======================
# 4. Training function
# =======================
def train_model(train_data, epochs=5):
    selector = ToolSelector(num_tools=3)
    optimizer = optim.Adam(selector.parameters(), lr=0.01)
    
    print("\nPrediction before training:")
    test_query = "basic concepts of artificial intelligence"
    idx, _ = selector.predict(encode_query(test_query))
    print(f"Query: '{test_query}'")
    print(f"Predicted tool: {selector.tool_names[idx]}")
    
    print("\nStart training...")
    for epoch in range(epochs):
        total_loss = 0
        tau = max(0.5, 1.0 - epoch * 0.1)
        
        for step, (query, target_tool) in enumerate(train_data):
            x = encode_query(query)
            _, logits = selector(x, tau=tau)
            
            tool_idx = selector.tool_names.index(target_tool)
            target = torch.zeros_like(logits)
            target[0, tool_idx] = 1.0
            
            loss = nn.functional.cross_entropy(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 5 == 0 or step == len(train_data) - 1:
                probs = torch.softmax(logits, dim=-1).detach().numpy()[0]
                print(
                    f"[Epoch {epoch}, Step {step}] Query='{query}' | "
                    f"Academic:{probs[0]:.3f} News:{probs[1]:.3f} Wiki:{probs[2]:.3f} | "
                    f"Loss: {loss.item():.4f} | τ={tau:.2f}"
                )
        
        print(f"Epoch {epoch} average loss: {total_loss / len(train_data):.4f}")
    
    return selector

# =======================
# 5. Main program
# =======================
if __name__ == "__main__":
    print("Gumbel-Softmax Tool Selector Demo (FAISS version)")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Initializing FAISS text encoder...")
    text_encoder = FAISSTextEncoder()
    
    train_data = [
        ("latest deep learning research methods", "academic"),
        ("today AI company stock prices", "news"),
        ("what is a knowledge graph", "wiki"),
    ]
    
    selector = train_model(train_data, epochs=5)
