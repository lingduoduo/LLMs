import os
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding

# --- Configure Alibaba Cloud DashScope API Key ---
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# Base URL for DashScope OpenAI-compatible mode
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# --- Initialize global LlamaIndex settings ---
Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base=DASHSCOPE_BASE_URL,
    api_key=DASHSCOPE_API_KEY,
    is_chat_model=True,
    temperature=0.1,
)

Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v4",
    api_key=DASHSCOPE_API_KEY,
)


def demonstrate_sliding_window_splitter(documents, chunk_size, chunk_overlap):
    """
    Demonstrate sliding-window chunking in LlamaIndex while preserving sentence boundaries.

    Args:
        documents (list[Document]): Documents to be split.
        chunk_size (int): Target number of tokens per chunk.
        chunk_overlap (int): Number of overlapping tokens between adjacent chunks.
    """
    print(f"\n{'=' * 50}")
    print("Demonstrating [Sliding Window Chunking]...")
    print(f"Chunk size (chunk_size): {chunk_size}")
    print(f"Overlap size (chunk_overlap): {chunk_overlap}")
    print(f"{'=' * 50}\n")

    # --- Step 1: Create the splitter ---
    # SentenceSplitter prioritizes preserving complete sentences before enforcing size limits
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # --- Step 2: Perform the split ---
    # Generate nodes (chunks) from the documents
    nodes = splitter.get_nodes_from_documents(documents)

    # --- Step 3: Print the chunks to demonstrate overlap ---
    print("\n--- Raw chunks generated after splitting ---")
    print(f"The document was split into {len(nodes)} chunks.")

    for i, node in enumerate(nodes, 1):
        content = node.get_content().strip()
        print(f"\n[Chunk {i}] (Length: {len(content)} characters):")
        print("-" * 50)
        print(f'Content:\n"{content}"')
        print("-" * 50)

    # --- Simple analysis: observe overlap between adjacent chunks ---
    print("\n--- Key Observation: Overlap between adjacent chunks ---")
    if len(nodes) > 1:
        # To better illustrate overlap, only show the overlapping parts
        # Since sentence-level splitting is used, the overlap consists of full sentences
        overlap_end_chunk1 = nodes[0].get_content()[-chunk_overlap:].strip()
        overlap_start_chunk2 = nodes[1].get_content()[:chunk_overlap].strip()

        print(f'End of Chunk 1 ({chunk_overlap} characters): "...{overlap_end_chunk1}"')
        print(f'Start of Chunk 2 ({chunk_overlap} characters): "{overlap_start_chunk2}..."')
        print(
            "\nAs you can see, the end of Chunk 1 overlaps with the start of Chunk 2. "
            "This is the effect of `chunk_overlap`."
        )
    else:
        print(
            "The document is too short to generate multiple chunks. "
            "Please use a longer document to observe the overlap effect."
        )

    print("\nSliding window chunking demonstration completed.")
    print(f"{'=' * 50}\n")


# --- Example document (contains multiple sentences) ---
# Use a longer, continuous document to better demonstrate splitting and overlap
documents = [
    Document(
        text="""
        LlamaIndex is a data framework for building LLM applications.
        It provides a set of tools that help developers connect private data with
        large language models (LLMs), enabling use cases such as question answering
        and Retrieval-Augmented Generation (RAG).
        LlamaIndex supports multiple data sources, including PDFs, databases, and APIs.

        Its core concepts include document loaders, node parsers, indexes, and query engines.
        Document loaders ingest data from various formats and sources into LlamaIndex.
        Node parsers then break loaded documents into smaller, more manageable units called nodes.
        These nodes are typically sentences or paragraphs, depending on the parsing strategy.
        Indexes are data structures built on top of these nodes to enable efficient storage
        and retrieval, usually involving vector embeddings for semantic search.
        """
    )
]

# --- Run the sliding window chunking demonstration ---
# Adjust chunk_size and chunk_overlap to observe different behaviors
demonstrate_sliding_window_splitter(
    documents,
    chunk_size=150,
    chunk_overlap=50,
)
