import os
from typing import List
from dataclasses import dataclass
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import dotenv

dotenv.load_dotenv()

@dataclass
class SourceInfo:
    """Information about a data source."""
    url: str
    title: str
    content: str
    timestamp: datetime


class SimpleRAGSystem:
    """A simple LangChain 1.x (LCEL) RAG system with source display."""

    def __init__(self, openai_api_key: str = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please set it as an environment variable.")

        # Embeddings + LLM
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.llm = OpenAI(
            model="gpt-3.5-turbo-instruct",
            openai_api_key=self.api_key,
            temperature=0,
        )

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Vector store + retriever + LCEL chain
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

        # Structured prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a fact-checking expert. Please answer the question based on the provided context.

Context:
{context}

Question: {question}

Please answer in the following format:
Verification Result: [True/False/Uncertain]
Confidence: [0-100%]
Reasoning: [Explain your reasoning in detail]
Evidence: [Quote the specific supporting evidence]
""".strip(),
        )

    def get_knowledge_sources(self) -> List[SourceInfo]:
        """Get knowledge sources (predefined content; no web crawling)."""
        now = datetime.now()
        return [
            SourceInfo(
                url="https://zh.wikipedia.org/wiki/人工智能",
                title="Artificial Intelligence - Wikipedia",
                content="""
Artificial intelligence (AI) is a branch of computer science focused on creating machines and software
that can perform tasks typically requiring human intelligence.

Major areas of AI include:
1. Machine learning
2. Deep learning
3. Natural language processing
4. Computer vision
5. Expert systems

There is significant debate about whether AI will surpass human intelligence within the next decade.
Some argue AGI may take decades; others believe breakthroughs could come sooner.
Current AI excels at specific tasks but remains far from general intelligence.
""",
                timestamp=now,
            ),
            SourceInfo(
                url="https://zh.wikipedia.org/wiki/机器学习",
                title="Machine Learning - Wikipedia",
                content="""
Machine learning is a branch of AI focused on algorithms that learn from data and improve over time.

Relationship:
- AI is broader
- Machine learning is one major approach to AI
- Deep learning is a subfield of machine learning

Hierarchy: AI > Machine Learning > Deep Learning

Types:
1. Supervised learning
2. Unsupervised learning
3. Reinforcement learning
""",
                timestamp=now,
            ),
            SourceInfo(
                url="https://zh.wikipedia.org/wiki/深度学习",
                title="Deep Learning - Wikipedia",
                content="""
Deep learning is a subfield of machine learning that uses multi-layer neural networks.

Hierarchy:
- AI is the broadest
- Machine learning is a branch of AI
- Deep learning is a specialized area of machine learning
""",
                timestamp=now,
            ),
            SourceInfo(
                url="https://en.wikipedia.org/wiki/Python_(programming_language)",
                title="Python - Wikipedia",
                content="""
Python is a high-level programming language known for its readable syntax.

Python in data science:
Python is one of the most popular languages in data science.
Surveys often report high usage among data scientists (e.g., Kaggle, Stack Overflow).
Python’s advantages include a rich library ecosystem and strong community support.
""",
                timestamp=now,
            ),
        ]

    def build_knowledge_base(self):
        """Build the FAISS knowledge base and LCEL chain."""
        print("🔧 Building the knowledge base...")

        sources = self.get_knowledge_sources()

        documents: List[Document] = []
        for source in sources:
            chunks = self.text_splitter.split_text(source.content)
            for chunk in chunks:
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": source.url,
                            "title": source.title,
                            "timestamp": source.timestamp.isoformat(),
                        },
                    )
                )

        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(d.page_content for d in docs)

        # LCEL RAG chain: question -> retrieve docs -> format -> prompt -> llm -> string
        self.qa_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        print(f"✅ Knowledge base built successfully, containing {len(documents)} document chunks")

    def ask_question(self, question: str):
        """Ask a question, print answer and sources."""
        if not self.qa_chain or not self.retriever:
            raise ValueError("Knowledge base not initialized. Call build_knowledge_base() first.")

        print(f"❓ Question: {question}")
        print("🤔 Thinking...")

        answer = self.qa_chain.invoke(question)
        source_docs = self.retriever.invoke(question)

        print("🤖 AI Answer:")
        print(answer)

        if source_docs:
            print("\n📚 Evidence Sources:")
            for i, doc in enumerate(source_docs, 1):
                print(f"   {i}. {doc.metadata.get('title', 'Unknown Source')}")
                print(f"      🔗 {doc.metadata.get('source', 'No Link')}")
                print(f"      📄 Snippet: {doc.page_content[:200]}...")
                print()

        return answer, source_docs


def main():
    print("=== Simple LangChain (LCEL) RAG System Demo ===\n")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Please set the OPENAI_API_KEY environment variable")
        return

    print("🚀 Initializing RAG system...")
    rag_system = SimpleRAGSystem(openai_api_key=openai_api_key)

    rag_system.build_knowledge_base()

    questions = [
        "Will artificial intelligence surpass human intelligence within the next decade?",
        "Is deep learning a subfield of machine learning?",
        "Is Python the most popular language in data science?",
    ]

    print("\n=== Starting Q&A Demo ===\n")
    for i, q in enumerate(questions, 1):
        print(f"{'=' * 60}")
        print(f"Question {i}")
        print(f"{'=' * 60}")
        try:
            rag_system.ask_question(q)
        except Exception as e:
            print(f"❌ Error: {e}")
        print(f"{'=' * 60}\n")

    print("✨ Demo completed!")


if __name__ == "__main__":
    main()
