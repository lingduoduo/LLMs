"""
CRAG (Corrective Retrieval-Augmented Generation) - LangChain Version
Pipeline: Retrieve (local) -> Relevance Grade -> Correct (query rewrite + web search) -> Fuse -> Answer
"""

import os
import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Optional

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Tavily tool (LangChain integration)
from langchain_community.tools.tavily_search import TavilySearchResults


# =========================
# Config + Models
# =========================
@dataclass
class CRAGConfig:
    # Local docs
    docs_dir: str = "./data"
    top_k: int = 4

    # LLMs
    answer_model: str = "gpt-4o-mini"
    grader_model: str = "gpt-4o-mini"
    rewrite_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_model: str = "text-embedding-3-large"

    # Web search
    tavily_max_results: int = 3

    # CRAG trigger
    # If ANY retrieved chunk graded "no" -> trigger web correction (matches your original logic)
    trigger_web_if_any_no: bool = True


def load_config_from_env() -> Tuple[str, Optional[str], Optional[str]]:
    """
    Loads environment variables.
    Supports:
      - OPENAI_API_KEY (required)
      - OPENAI_BASE_URL (optional, for OpenAI-compatible endpoints)
      - TAVILY_API_KEY (optional, enables web correction)
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set. Put it in your .env and rerun.")
    openai_base_url = os.getenv("OPENAI_BASE_URL")  # optional
    tavily_api_key = os.getenv("TAVILY_API_KEY")    # optional
    return openai_api_key, openai_base_url, tavily_api_key


def build_llm(model: str, openai_api_key: str, openai_base_url: Optional[str] = None) -> ChatOpenAI:
    kwargs = {"model": model, "api_key": openai_api_key}
    if openai_base_url:
        kwargs["base_url"] = openai_base_url
    return ChatOpenAI(**kwargs)


def load_documents_simple(docs_dir: str) -> List[Document]:
    """
    Minimal directory loader (no LlamaIndex).
    Loads .txt/.md by default. Extend if you need PDF parsing, etc.
    """
    docs: List[Document] = []
    if not os.path.isdir(docs_dir):
        raise ValueError(f"docs_dir not found: {docs_dir}")

    for root, _, files in os.walk(docs_dir):
        for fn in files:
            if not fn.lower().endswith((".txt", ".md")):
                continue
            path = os.path.join(root, fn)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": path}))
    return docs


def build_vectorstore(
    docs: List[Document],
    openai_api_key: str,
    openai_base_url: Optional[str],
    embedding_model: str,
) -> FAISS:
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=openai_api_key,
        base_url=openai_base_url,  # harmless if None
    )
    return FAISS.from_documents(docs, embeddings)


# =========================
# CRAG Prompts
# =========================
RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict-but-not-overly-strict relevance grader for RAG retrieval. "
            "Your job is to filter out obviously irrelevant chunks.",
        ),
        (
            "user",
            "User question:\n{question}\n\n"
            "Retrieved chunk:\n{chunk}\n\n"
            "Answer ONLY 'yes' if the chunk is relevant, otherwise answer ONLY 'no'.",
        ),
    ]
)

REWRITE_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You rewrite queries to improve web search recall and precision."),
        ("user", "Rewrite this query for better web search results. Return ONLY the rewritten query:\n{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer using the provided context. "
            "If the context is insufficient, say so explicitly.",
        ),
        (
            "user",
            "Question:\n{question}\n\n"
            "Context:\n{context}\n\n"
            "Write a concise, factual answer.",
        ),
    ]
)


# =========================
# CRAG Core Logic
# =========================
async def grade_relevance(
    llm: ChatOpenAI,
    question: str,
    chunks: List[str],
) -> List[str]:
    """
    Returns list of 'yes'/'no' per chunk.
    """
    chain = RELEVANCE_PROMPT | llm | StrOutputParser()
    # grade sequentially to keep it simple/stable; can parallelize if desired
    results = []
    for c in chunks:
        r = (await chain.ainvoke({"question": question, "chunk": c})).strip().lower()
        results.append("yes" if r.startswith("y") else "no")
    return results


async def rewrite_query(llm: ChatOpenAI, question: str) -> str:
    chain = REWRITE_QUERY_PROMPT | llm | StrOutputParser()
    return (await chain.ainvoke({"question": question})).strip()


def should_trigger_web(relevance_labels: List[str], trigger_if_any_no: bool) -> bool:
    if not relevance_labels:
        return True
    if trigger_if_any_no:
        return "no" in relevance_labels
    # alternative: trigger only if majority are no
    return relevance_labels.count("no") > relevance_labels.count("yes")


def fuse_context(local_chunks: List[str], web_snippets: List[str]) -> str:
    parts = []
    if local_chunks:
        parts.append("LOCAL CONTEXT:\n" + "\n\n".join(local_chunks))
    if web_snippets:
        parts.append("WEB SEARCH CONTEXT:\n" + "\n\n".join(web_snippets))
    return "\n\n".join(parts).strip()


async def answer_question(llm: ChatOpenAI, question: str, context: str) -> str:
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    return (await chain.ainvoke({"question": question, "context": context})).strip()


async def crag_query(
    question: str,
    vectorstore: FAISS,
    cfg: CRAGConfig,
    llm_answer: ChatOpenAI,
    llm_grader: ChatOpenAI,
    llm_rewrite: ChatOpenAI,
    tavily_tool: Optional[TavilySearchResults] = None,
) -> dict:
    """
    Full CRAG run: local retrieve -> grade -> optional web correction -> fuse -> answer
    Returns a structured dict for inspection.
    """
    # Step 1: Local retrieve
    retriever = vectorstore.as_retriever(search_kwargs={"k": cfg.top_k})
    retrieved_docs: List[Document] = retriever.invoke(question)
    retrieved_chunks = [d.page_content for d in retrieved_docs]

    # Step 2: Relevance grading
    relevance = await grade_relevance(llm_grader, question, retrieved_chunks)

    # Step 3: Keep relevant local text
    relevant_local = [chunk for chunk, lab in zip(retrieved_chunks, relevance) if lab == "yes"]

    # Step 4: Correct (query rewrite + web search) if needed
    web_snippets: List[str] = []
    transformed_query: Optional[str] = None

    if tavily_tool and should_trigger_web(relevance, cfg.trigger_web_if_any_no):
        transformed_query = await rewrite_query(llm_rewrite, question)
        try:
            web_results = tavily_tool.invoke({"query": transformed_query, "max_results": cfg.tavily_max_results})
            # web_results is typically a list of dicts: {"title","url","content"} depending on version
            for item in web_results:
                content = item.get("content") or item.get("snippet") or ""
                title = item.get("title") or ""
                if title and content:
                    web_snippets.append(f"{title}\n{content}")
                elif content:
                    web_snippets.append(content)
        except Exception as e:
            web_snippets = [f"[Web search failed: {e}]"]

    # Step 5: Fuse + Answer
    fused = fuse_context(relevant_local, web_snippets)
    if not fused:
        final_answer = "Sorry, I could not find relevant information to answer your question."
    else:
        final_answer = await answer_question(llm_answer, question, fused)

    return {
        "question": question,
        "retrieved_count": len(retrieved_docs),
        "relevance_labels": relevance,
        "kept_local_chunks": len(relevant_local),
        "transformed_query": transformed_query,
        "used_web": bool(web_snippets),
        "final_answer": final_answer,
    }


# =========================
# Demo
# =========================
async def run_demo():
    print("Starting CRAG demo (LangChain version)...")

    openai_api_key, openai_base_url, tavily_api_key = load_config_from_env()

    cfg = CRAGConfig()

    # Build models
    llm_answer = build_llm(cfg.answer_model, openai_api_key, openai_base_url)
    llm_grader = build_llm(cfg.grader_model, openai_api_key, openai_base_url)
    llm_rewrite = build_llm(cfg.rewrite_model, openai_api_key, openai_base_url)

    # Build web tool (optional)
    tavily_tool = TavilySearchResults(api_key=tavily_api_key) if tavily_api_key else None
    if tavily_tool:
        print("Tavily web search enabled.")
    else:
        print("Tavily web search NOT enabled (set TAVILY_API_KEY to enable correction).")

    # Load docs + build vectorstore
    docs = load_documents_simple(cfg.docs_dir)
    print(f"Loaded {len(docs)} documents from: {cfg.docs_dir}")

    vectorstore = build_vectorstore(docs, openai_api_key, openai_base_url, cfg.embedding_model)
    print("Vector store built.")

    # Queries (English-only demo, replace as needed)
    test_queries = [
        "What was the U.S. federal budget deficit in fiscal year 2020?",
        "Which federal programs contributed most to increased government costs during the COVID-19 pandemic?",
        "Why did net operating cost increase more than the budget deficit in FY 2020?",
        "What was the debt-to-GDP ratio at the end of FY 2020, and why is it considered unsustainable?",
    ]

    for i, q in enumerate(test_queries, 1):
        print("\n" + "=" * 70)
        print(f"Test Query {i}: {q}")
        print("=" * 70)

        out = await crag_query(
            question=q,
            vectorstore=vectorstore,
            cfg=cfg,
            llm_answer=llm_answer,
            llm_grader=llm_grader,
            llm_rewrite=llm_rewrite,
            tavily_tool=tavily_tool,
        )

        print("Retrieved:", out["retrieved_count"])
        print("Relevance labels:", out["relevance_labels"])
        print("Kept local chunks:", out["kept_local_chunks"])
        if out["transformed_query"]:
            print("Rewritten query:", out["transformed_query"])
        print("Used web:", out["used_web"])
        print("\nAnswer:\n", out["final_answer"])


if __name__ == "__main__":
    asyncio.run(run_demo())

'''
(llm_clean)  🐍 llm_clean  linghuang@Mac  ~/Git/LLMs/LLM-RAG/34-CRAG/34   rag-optimization  /Users/linghuang/miniconda3/envs/llm_clean/bin/python /Users/linghuang/Git/
LLMs/LLM-RAG/34-CRAG/34/crag.py
Starting CRAG demo (LangChain version)...
Tavily web search NOT enabled (set TAVILY_API_KEY to enable correction).
Loaded 1 documents from: ./data
Vector store built.

======================================================================
Test Query 1: What was the U.S. federal budget deficit in fiscal year 2020?
======================================================================
Retrieved: 1
Relevance labels: ['yes']
Kept local chunks: 1
Used web: False

Answer:
 The U.S. federal budget deficit in fiscal year 2020 was $3,131.9 billion (or approximately $3.1 trillion).

======================================================================
Test Query 2: Which federal programs contributed most to increased government costs during the COVID-19 pandemic?
======================================================================
Retrieved: 1
Relevance labels: ['yes']
Kept local chunks: 1
Used web: False

Answer:
 The federal programs that contributed most to increased government costs during the COVID-19 pandemic include:

1. **Paycheck Protection Program (PPP)** - This program resulted in a $559.1 billion increase in net costs, primarily due to loan subsidy costs related to loans and loan forgiveness for eligible business expenses.
   
2. **Unemployment Insurance (UI) program administered by the Department of Labor (DOL)** - Increased costs amounted to $452.7 billion, largely attributable to expanded unemployment benefits authorized by the CARES Act.

3. **Economic Impact Payments (EIPs) issued by the Treasury** - These payments to individuals resulted in a $405.0 billion increase in Treasury net costs.

4. **Public Health and Social Services Emergency Fund (PHSSEF) administered by the Department of Health and Human Services (HHS)** - This program saw a net cost increase of $124.8 billion, primarily due to funding for health care providers and pandemic response efforts.

These programs accounted for significant expenditures, contributing to a substantial rise in the federal government's overall financial costs during the fiscal year 2020.

======================================================================
Test Query 3: Why did net operating cost increase more than the budget deficit in FY 2020?
======================================================================
Retrieved: 1
Relevance labels: ['yes']
Kept local chunks: 1
Used web: False

Answer:
 The net operating cost increased more than the budget deficit in FY 2020 primarily due to a combination of factors that included accrued costs that are recognized in net operating cost but not in the budget deficit. Specifically, there was an increase of $2.3 trillion in net cost mainly driven by significant increases in federal employee and veteran benefits liabilities, along with other pandemic-related costs. In contrast, the budget deficit increased by $2.1 trillion, resulting in a difference of $696.9 billion. This difference highlights the effect of accruing costs that were incurred but not necessarily paid, which are factored into net operating cost.

======================================================================
Test Query 4: What was the debt-to-GDP ratio at the end of FY 2020, and why is it considered unsustainable?
======================================================================
Retrieved: 1
Relevance labels: ['yes']
Kept local chunks: 1
Used web: False

Answer:
 At the end of FY 2020, the debt-to-GDP ratio was 100 percent. This ratio is considered unsustainable because it indicates that the government’s debt equals the total economic output of the country, suggesting a potential inability to manage and repay that debt in the long term. Projections indicate that if current fiscal policies continue, this ratio could rise significantly, leading to a continuously increasing debt burden relative to the economy, which is unsustainable over time. The report estimates that making fiscal policy sustainable would require significant reductions in primary deficits, amounting to about 5.4 percent of GDP over the next 75 years.
(llm_clean)  🐍 llm_clean  linghuang@Mac  ~/Git/LLMs/LLM-RAG/34-CRAG/34   rag-optimization  
'''