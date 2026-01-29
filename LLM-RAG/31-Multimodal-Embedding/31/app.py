import os
import io
import base64
from typing import List, Tuple

import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# =========================
# Data (edit as needed)
# =========================
PRODUCT_DATABASE = [
    "Light blue shirt, cotton fabric, suitable for summer wear, sizes from S to XL",
    "Red dress, cotton fabric, suitable for summer wear, sizes from S to XL",
    "White T-shirt, 100% cotton, crew neck short sleeves, versatile style",
    "Black leather shoes, genuine leather, business formal, durable and slip-resistant",
    "Sneakers, lightweight and breathable, suitable for running and fitness, multiple colors available",
]


# =========================
# Helpers
# =========================
def load_config() -> str:
    """Load environment variables and return OpenAI API key."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Put it in .env and restart Streamlit.")
    return api_key


def pil_to_jpeg_bytes(image: Image.Image) -> bytes:
    """Convert a PIL image to JPEG bytes (handles RGBA)."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def image_to_text(client: OpenAI, image_bytes: bytes, vision_model: str = "gpt-4o-mini") -> str:
    """Use OpenAI vision to generate a structured product description from an image."""
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    resp = client.chat.completions.create(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {
                        "type": "text",
                        "text": (
                            "You are an e-commerce product tagger.\n"
                            "Describe the product concisely for retrieval.\n"
                            "Return 5-10 bullet points covering:\n"
                            "- category/type\n"
                            "- color(s)\n"
                            "- material(s)\n"
                            "- key features\n"
                            "- typical use cases\n"
                            "Avoid brand names unless visible."
                        ),
                    },
                ],
            }
        ],
    )
    return resp.choices[0].message.content.strip()


@st.cache_resource
def setup_vector_store(openai_api_key: str, embedding_model: str = "text-embedding-3-large") -> FAISS:
    """Build and cache a FAISS vector store over the product database."""
    docs = [Document(page_content=text, metadata={"product_id": i}) for i, text in enumerate(PRODUCT_DATABASE)]
    embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=openai_api_key)
    return FAISS.from_documents(docs, embeddings)


def search_similar_products(
    query_text: str,
    vectorstore: FAISS,
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Retrieve similar products with scores.

    Note: FAISS returns a distance score (lower usually means more similar).
    """
    results = vectorstore.similarity_search_with_score(query_text, k=top_k)
    return [(doc.page_content, score) for doc, score in results]


# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(page_title="Product Retrieval", layout="centered")
    st.title("Product Image Retrieval (OpenAI + LangChain + FAISS)")

    # --- Config ---
    try:
        api_key = load_config()
    except Exception as e:
        st.error(str(e))
        st.stop()

    client = OpenAI(api_key=api_key)

    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Top-K results", min_value=1, max_value=10, value=3, step=1)
    vision_model = st.sidebar.selectbox("Vision model", ["gpt-4o-mini", "gpt-4o"], index=0)
    embedding_model = st.sidebar.selectbox("Embedding model", ["text-embedding-3-large", "text-embedding-3-small"], index=0)

    # Build vector store once (cached)
    vectorstore = setup_vector_store(api_key, embedding_model=embedding_model)

    st.write("### Option A: Upload an image")
    uploaded_file = st.file_uploader("Choose a product image", type=["jpg", "jpeg", "png"])

    st.write("### Option B: Or type a text query")
    text_query = st.text_input("Example: 'blue high-waist jeans, slim fit'")

    # --- Run retrieval ---
    if uploaded_file is None and not text_query.strip():
        st.info("Upload an image or enter a text query to search.")
        with st.expander("View Product Database"):
            for i, p in enumerate(PRODUCT_DATABASE, start=1):
                st.write(f"{i}. {p}")
        return

    query_for_retrieval = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", width=320)
        image_bytes = pil_to_jpeg_bytes(image)

        if st.button("Analyze image and search"):
            try:
                with st.spinner("Generating image description..."):
                    desc = image_to_text(client, image_bytes, vision_model=vision_model)

                st.subheader("Image Description")
                st.write(desc)

                query_for_retrieval = desc

            except Exception as e:
                st.error("Vision step failed:")
                st.exception(e)
                st.stop()

    # If text query provided, it can run immediately (or override if no image-run)
    if text_query.strip():
        query_for_retrieval = text_query.strip()

    if query_for_retrieval:
        try:
            with st.spinner("Searching similar products..."):
                results = search_similar_products(query_for_retrieval, vectorstore, top_k=top_k)

            st.subheader("Retrieval Results")
            if not results:
                st.write("No similar products found.")
            else:
                for i, (text, score) in enumerate(results, start=1):
                    st.write(f"{i}. **Distance:** {score:.4f} — {text}")

            with st.expander("View Product Database"):
                for i, p in enumerate(PRODUCT_DATABASE, start=1):
                    st.write(f"{i}. {p}")

        except Exception as e:
            st.error("Retrieval step failed:")
            st.exception(e)


if __name__ == "__main__":
    main()
