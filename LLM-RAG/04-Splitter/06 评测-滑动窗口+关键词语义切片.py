import os
import re
import textwrap
from typing import List, Callable
from pydantic import Field

from llama_index.core import Settings, Document
from llama_index.core.node_parser import (
    NodeParser,
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core.utils import get_tokenizer

# --------------------------------------------------
# Global configuration
# --------------------------------------------------
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# --------------------------------------------------
# Hybrid node parser definition (with detailed logging)
# --------------------------------------------------
class HybridNodeParser(NodeParser):
    """
    A hybrid node parser that combines:
    1) Semantic splitting (coarse-grained, topic-aware)
    2) Sentence-based sliding window splitting (fine-grained, size-controlled)

    Large semantic chunks are automatically split again using a sliding window
    strategy to ensure chunk size constraints.
    """

    primary_parser: NodeParser
    secondary_parser: NodeParser
    max_chunk_size: int = 1024
    tokenizer: Callable = Field(default_factory=get_tokenizer, exclude=True)

    def _parse_nodes(self, documents: List[Document], **kwargs) -> List[Document]:
        print("--- Starting [Hybrid Chunking] ---")

        # Step 1: Semantic splitting
        primary_nodes = self.primary_parser.get_nodes_from_documents(documents)

        print(f"\n{'=' * 25} Step 1: Semantic Splitting {'=' * 25}")
        print(f"Generated {len(primary_nodes)} semantic chunks.")

        for i, p_node in enumerate(primary_nodes, 1):
            size = len(self.tokenizer(p_node.get_content()))
            print(f"\n[Semantic Chunk {i}] (Size: {size} tokens)")
            print("-" * 60)
            print(textwrap.indent(p_node.get_content().strip(), "  "))
            print("-" * 60)

        # Step 2: Size check and secondary splitting
        print(f"\n{'=' * 25} Step 2: Size Check & Secondary Splitting {'=' * 25}")
        final_nodes = []

        for i, node in enumerate(primary_nodes, 1):
            node_size = len(self.tokenizer(node.get_content()))
            print(f"\n>>> Checking [Semantic Chunk {i}] (Size: {node_size} tokens)")

            if node_size <= self.max_chunk_size:
                print(f"  └── Accepted (<= {self.max_chunk_size} tokens).")
                final_nodes.append(node)
            else:
                print(
                    f"  └── Too large (> {self.max_chunk_size} tokens). "
                    "Applying sliding window splitting."
                )

                print("\n      [Original content to be split]")
                print("      " + "-" * 50)
                print(textwrap.indent(node.get_content().strip(), "      | "))
                print("      " + "-" * 50)

                sub_nodes = self.secondary_parser.get_nodes_from_documents(
                    [Document(text=node.get_content())]
                )

                print(
                    f"\n      [Secondary split result]: "
                    f"{len(sub_nodes)} overlapping sub-chunks generated."
                )

                for j, s_node in enumerate(sub_nodes, 1):
                    sub_size = len(self.tokenizer(s_node.get_content()))
                    print(f"\n        [Sub-chunk {i}.{j}] (Size: {sub_size} tokens)")
                    print("        " + "-" * 40)
                    print(textwrap.indent(s_node.get_content().strip(), "        | "))
                    print("        " + "-" * 40)

                final_nodes.extend(sub_nodes)

        print("\n--- [Hybrid Chunking Completed] ---")
        return final_nodes

    @classmethod
    def from_defaults(cls, **kwargs):
        raise NotImplementedError(
            "Please instantiate this class directly; do not use from_defaults()."
        )


# --------------------------------------------------
# Initialize LlamaIndex global settings
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

# --------------------------------------------------
# English sentence tokenizer
# --------------------------------------------------
def english_sentence_tokenizer(text: str) -> list[str]:
    sentences = re.findall(r"[^.!?\n]+[.!?\n]?", text)
    return [s.strip() for s in sentences if s.strip()]

# --------------------------------------------------
# Prepare a long document
# --------------------------------------------------
long_document = Document(
    text="""
    LlamaIndex is a data framework for building, evaluating, and deploying advanced
    Retrieval-Augmented Generation (RAG) applications. Its core mission is to help
    developers seamlessly connect private or domain-specific data with large language
    models (LLMs). By providing data connectors, indexing structures, query engines,
    and evaluation tools, LlamaIndex significantly simplifies the entire pipeline
    from raw data to production-ready RAG systems. It supports hundreds of data sources,
    including PDF files, databases, Notion pages, and various APIs, ensuring flexible
    data ingestion.

    From an architectural perspective, LlamaIndex is designed around modularity and
    extensibility. Its core components include Document Loaders, Node Parsers, Indices,
    Retrievers, and Query Engines. Document loaders ingest data from heterogeneous sources.
    Node parsers (such as the ones demonstrated here) decompose documents into smaller,
    more manageable units called nodes, which serve as the fundamental building blocks
    for indexing. Index structures—especially vector indices—leverage embedding models
    to transform nodes into high-dimensional vectors, enabling efficient semantic search
    and retrieval. Retrievers identify the most relevant nodes for a given query, and
    query engines orchestrate retrieval and LLM reasoning to produce context-aware answers.
    This architecture allows developers to assemble custom RAG pipelines in a highly
    composable manner.

    Evaluation is another major strength of LlamaIndex. It provides a comprehensive set
    of tools to quantify RAG performance, including answer faithfulness, relevance, and
    retrieval precision and recall. These metrics enable systematic debugging and
    optimization, moving beyond intuition-driven iteration toward data-driven improvement.
    """
)

# --------------------------------------------------
# Instantiate base parsers
# --------------------------------------------------
semantic_parser = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    sentence_splitter=english_sentence_tokenizer,
    embed_model=Settings.embed_model,
)

window_parser = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=50,
)

# --------------------------------------------------
# Instantiate and run the hybrid parser
# --------------------------------------------------
hybrid_parser = HybridNodeParser(
    primary_parser=semantic_parser,
    secondary_parser=window_parser,
    max_chunk_size=300,
    tokenizer=get_tokenizer(),
)

final_nodes = hybrid_parser.get_nodes_from_documents([long_document])

# --------------------------------------------------
# Print final results
# --------------------------------------------------
print(f"\n{'=' * 25} Final Generated Chunks {'=' * 25}")
print(f"Total number of chunks: {len(final_nodes)}")

for i, node in enumerate(final_nodes, 1):
    content = node.get_content().strip()
    print(f"\n[Final Chunk {i}]")
    print("-" * 50)
    print(textwrap.indent(content, "  "))
    print("-" * 50)
