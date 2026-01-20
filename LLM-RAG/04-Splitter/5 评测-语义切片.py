import os
import re
import numpy as np
import matplotlib.pyplot as plt
from llama_index.core import Settings, Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core.utils import get_tokenizer

# --------------------------------------------------
# 1. Global configuration
# --------------------------------------------------
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# --------------------------------------------------
# 2. Initialize LlamaIndex global settings
# --------------------------------------------------
Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base=DASHSCOPE_BASE_URL,
    api_key=DASHSCOPE_API_KEY,
    is_chat_model=True,
    temperature=0.1,
)

Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v2",
    api_key=DASHSCOPE_API_KEY,
    batch_size=10,
)

llama_tokenizer = get_tokenizer()

# --------------------------------------------------
# 3. English sentence tokenizer
# --------------------------------------------------
# This function is directly used by SemanticSplitterNodeParser
def english_sentence_tokenizer(text: str) -> list[str]:
    sentences = re.findall(r"[^.!?\n]+[.!?\n]?", text)
    return [s.strip() for s in sentences if s.strip()]

# --------------------------------------------------
# 4. Prepare a long English document
# --------------------------------------------------
long_text = (
    "Large Language Models (LLMs) represent a revolutionary advancement in artificial intelligence. "
    "Their development can be traced back to early research in neural networks and natural language processing. "
    "Early models such as Recurrent Neural Networks (RNNs) suffered from the vanishing gradient problem when handling long text sequences. "
    "Later, Long Short-Term Memory networks (LSTMs) partially mitigated this issue but still had inherent limitations.\n\n"
    "A major breakthrough came in 2017 with the introduction of the Transformer architecture. "
    "Transformers rely entirely on self-attention mechanisms, enabling parallel processing of text sequences and significantly improving training efficiency and performance. "
    "This innovation laid the foundation for large-scale models such as BERT and the GPT family. "
    "BERT uses bidirectional encoders and excels at contextual understanding, while GPT adopts an autoregressive decoder and is particularly strong at text generation.\n\n"
    "Today, LLM applications have spread across many industries. "
    "They support intelligent question answering, content creation, and code generation, and demonstrate strong potential in healthcare, finance, and education. "
    "For example, in healthcare, LLMs can assist doctors by analyzing medical records and providing diagnostic suggestions. "
    "In education, they can offer personalized tutoring and learning resources.\n\n"
    "However, the development of LLMs also introduces significant challenges. "
    "Model bias, hallucinations, and high computational costs remain open problems. "
    "Future research directions include improving model efficiency, enhancing interpretability, and establishing responsible AI governance frameworks. "
    "Ensuring that these technologies are safe, fair, and widely accessible is a shared responsibility across the research community."
)

document = Document(text=long_text)

# --------------------------------------------------
# 5. Similarity visualization and semantic splitting
# --------------------------------------------------
def plot_similarity_and_chunks(splitter: SemanticSplitterNodeParser, title: str):
    # Sentence segmentation for visualization
    sentences = english_sentence_tokenizer(document.get_content())

    if len(sentences) < 2:
        print(f"Error: Only {len(sentences)} sentence(s) found. Cannot compute similarity.")
        return

    print(f"Generating embeddings for {len(sentences)} sentences...")
    embeddings = Settings.embed_model.get_text_embedding_batch(sentences, show_progress=True)

    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    similarities = [
        cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    threshold_value = np.percentile(similarities, splitter.breakpoint_percentile_threshold)
    print(f"Computed similarity threshold: {threshold_value:.4f}")

    # --- Visualization ---
    plt.figure(figsize=(12, 6))
    plt.plot(similarities, marker="o", linestyle="-", label="Adjacent sentence similarity")
    plt.axhline(
        y=threshold_value,
        linestyle="--",
        label=f"Breakpoint threshold ({splitter.breakpoint_percentile_threshold}th percentile)",
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Sentence boundary index")
    plt.ylabel("Cosine similarity")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Actual semantic splitting ---
    nodes = splitter.get_nodes_from_documents([document])

    print("\n--- Semantic Split Results ---")
    for i, node in enumerate(nodes):
        token_len = len(llama_tokenizer(node.get_content()))
        print(f"====== Node {i + 1} (Length: {token_len} tokens) ======")
        print(node.get_content().strip())
        print("-" * 30)

# --------------------------------------------------
# 6. Experiments with different thresholds
# --------------------------------------------------

# Experiment 1: Conservative threshold
print("=" * 20 + " Experiment 1: Conservative Threshold (95) " + "=" * 20)
conservative_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=Settings.embed_model,
    sentence_splitter=english_sentence_tokenizer,
)
plot_similarity_and_chunks(conservative_splitter, "Semantic Similarity & Breakpoints (95%)")

# Experiment 2: Aggressive threshold
print("\n" + "=" * 20 + " Experiment 2: Aggressive Threshold (5) " + "=" * 20)
aggressive_splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=5,
    embed_model=Settings.embed_model,
    sentence_splitter=english_sentence_tokenizer,
)
plot_similarity_and_chunks(aggressive_splitter, "Semantic Similarity & Breakpoints (5%)")
