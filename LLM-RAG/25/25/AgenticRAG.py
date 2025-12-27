import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain core
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

# LangChain + OpenAI (works with OpenAI-compatible endpoints)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

# Loaders / splitters
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Agent
from langchain.agents import AgentExecutor, create_react_agent


# -----------------------------
# 0) Environment / Keys
# -----------------------------
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY is not set. Add it to your .env file or export it in your shell."
    )
API_BASE = os.getenv("OPENAI_BASE_URL", "").strip() or None
print("Initializing the Agentic RAG system (LangChain)...")
if API_BASE:
    print("Using OPENAI_BASE_URL:", API_BASE)
else:
    print("Using default OpenAI base_url (https://api.openai.com/v1)")

# -----------------------------
# 1) Configure LangChain LLM + Embeddings
# -----------------------------
llm_kwargs = dict(
    api_key=openai_api_key,
    model="gpt-3.5-turbo",
    temperature=0,
)
emb_kwargs = dict(
    api_key=openai_api_key,
    # ✅ use a modern embedding model by default
    model="text-embedding-3-small",
)

# Only set base_url if provided
if API_BASE:
    llm_kwargs["base_url"] = API_BASE
    emb_kwargs["base_url"] = API_BASE

llm = ChatOpenAI(**llm_kwargs)
embeddings = OpenAIEmbeddings(**emb_kwargs)


def sanity_check_openai_compatible(llm, embeddings):
    # LLM check
    try:
        r = llm.invoke("Reply with exactly: OK")
        print("[SanityCheck] LLM OK:", getattr(r, "content", r))
    except Exception as e:
        raise RuntimeError(
            "[SanityCheck] LLM call failed. Check OPENAI_API_KEY and OPENAI_BASE_URL.\n"
            f"Underlying error: {e}"
        )

    # Embedding check (this is what FAISS indexing needs)
    try:
        v = embeddings.embed_query("hello")
        print("[SanityCheck] Embeddings OK. dim =", len(v))
    except Exception as e:
        raise RuntimeError(
            "[SanityCheck] Embedding call failed. This will prevent FAISS indexing.\n"
            "Check OPENAI_API_KEY and OPENAI_BASE_URL, and that the embedding model is supported.\n"
            f"Underlying error: {e}"
        )


sanity_check_openai_compatible(llm, embeddings)

# -----------------------------
# 2) Resolve paths robustly
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

file_list = [DATA_DIR / "crag.txt", DATA_DIR / "selfrag.txt", DATA_DIR / "kgrag.txt"]
name_list = ["c-rag", "self-rag", "kg-rag"]

print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", DATA_DIR)
for p in file_list:
    print("Looking for:", p, "| exists =", p.exists())

# -----------------------------
# 3) Build tools (vector lookup + summary) per doc
# -----------------------------
tools = []

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=80,
)

MAP_PROMPT = PromptTemplate.from_template(
    "You are summarizing a technical document chunk.\n"
    "Chunk:\n{chunk}\n\n"
    "Write a concise chunk summary focusing on key technical ideas:"
)

REDUCE_PROMPT = PromptTemplate.from_template(
    "You are writing a final technical summary from chunk summaries.\n"
    "Chunk summaries:\n{summaries}\n\n"
    "Write a coherent high-level summary (core ideas, contributions, conclusions), "
    "keeping it factual and grounded in the summaries:"
)


def load_and_split_txt(file_path: Path, doc_name: str) -> list[Document]:
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata = {**(d.metadata or {}), "source": str(file_path), "doc_name": doc_name}
    return splitter.split_documents(docs)


def build_vector_tool(doc_name: str, docs: list[Document]) -> Tool:
    vs = FAISS.from_documents(docs, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    def vector_lookup(query: str) -> str:
        retrieved = retriever.get_relevant_documents(query)
        if not retrieved:
            return f"No relevant passages found in {doc_name}."
        joined = "\n\n".join(
            f"[{i+1}] (source={d.metadata.get('source')})\n{d.page_content}"
            for i, d in enumerate(retrieved)
        )
        return (
            f"Top passages from {doc_name}:\n\n{joined}\n\n"
            "Instruction: Answer using ONLY the passages above. If missing, say you can't find it."
        )

    return Tool(
        name=f"{doc_name}_vector_tool",
        description=(
            f"Query specific technical details in the {doc_name} document "
            f"(methods, formulas, experimental settings). Input should be a natural language question."
        ),
        func=vector_lookup,
    )


def build_summary_tool(doc_name: str, docs: list[Document]) -> Tool:
    def summarize_doc(_: str = "") -> str:
        chunk_summaries = []
        for d in docs:
            chunk_summaries.append(llm.invoke(MAP_PROMPT.format(chunk=d.page_content)).content)
        combined = "\n".join(f"- {s}" for s in chunk_summaries)
        final = llm.invoke(REDUCE_PROMPT.format(summaries=combined)).content
        return (
            f"High-level summary of {doc_name}:\n{final}\n\n"
            "Note: This summary is grounded in chunk summaries of the document."
        )

    return Tool(
        name=f"{doc_name}_summary_tool",
        description=(
            f"Get a high-level overview of the {doc_name} document "
            f"(core ideas, contributions, conclusions). Input can be empty or any text."
        ),
        func=summarize_doc,
    )


for file_path, name in zip(file_list, name_list):
    print(f"\nProcessing document: {file_path}")

    if not file_path.exists():
        print(f"✗ Skipped (missing file): {file_path}")
        continue

    try:
        doc_chunks = load_and_split_txt(file_path, name)

        vector_tool = build_vector_tool(name, doc_chunks)
        summary_tool = build_summary_tool(name, doc_chunks)

        tools.extend([vector_tool, summary_tool])
        print(f"✓ Finished processing {file_path} (chunks={len(doc_chunks)})")

    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")

print(f"\nCreated {len(tools)} tools in total.")

if not tools:
    raise FileNotFoundError(
        f"No tools created. Put your txt files under: {DATA_DIR}\n"
        f"Expected: crag.txt, selfrag.txt, kgrag.txt"
    )

# -----------------------------
# 4) Initialize ReAct agent
# -----------------------------
react_prompt = PromptTemplate.from_template(
    "You are an expert in Retrieval-Augmented Generation (RAG).\n"
    "Use the provided tools to answer.\n"
    "Follow ReAct: Thought → Action → Observation → Answer.\n"
    "Only use tool results; do not fabricate information.\n\n"
    "TOOLS:\n{tools}\n\n"
    "Tool names: {tool_names}\n\n"
    "Question: {input}\n\n"
    "{agent_scratchpad}"
)

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("✓ Agent initialized successfully")


def main():
    print("\n=== Agentic RAG System Started (LangChain) ===")
    print("Available tools:")
    for t in tools:
        print(f"- {t.name}: {t.description}")
    print("\n" + "=" * 50 + "\n")

    queries = [
        "Compare the core technical approaches of c-rag and self-rag, and explain how they differ in reducing hallucinations.",
        "Extract the product-style introduction from the kg-rag document and rewrite it as a concise customer-facing description.",
        "Summarize the main advantages and suitable use cases of these three RAG methods.",
    ]

    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 50)
        try:
            result = agent_executor.invoke({"input": query})
            print("Answer:", result.get("output", result))
        except Exception as e:
            print(f"Query failed: {e}")
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
