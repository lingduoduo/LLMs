import os
import io
import base64

import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# -------------------------
# Data
# -------------------------
PRODUCT_DATABASE = [
    "Light blue shirt, cotton fabric, suitable for summer wear, sizes from S to XL",
    "Red dress, cotton fabric, suitable for summer wear, sizes from S to XL",
    "Blue jeans, high-waist design, slim fit, durable and wear-resistant",
    "White T-shirt, 100% cotton, crew neck, short sleeves, versatile style",
    "Black leather dress shoes, genuine leather, business formal, durable and slip-resistant",
    "Athletic sneakers, lightweight and breathable, suitable for running and workouts, multiple colors available",
    "Women's handbag, PU leather, large capacity, multiple internal compartments",
    "Men's wallet, top-grain cowhide leather, multiple card slots, classic black",
    "Sunglasses, polarized lenses, UV protection, metal frame",
    "Smartwatch, heart rate monitoring, step counter, waterproof design",
    "Wireless earbuds, Bluetooth 5.0, noise cancellation, long battery life",
]


# -------------------------
# Helpers (no Streamlit side effects)
# -------------------------
def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    fmt = (img.format or "JPEG").upper()
    # Some PIL images may have format=None even if filename is PNG/JPG; JPEG is fine for vision.
    img.save(buf, format=fmt if fmt in {"JPEG", "JPG", "PNG"} else "JPEG")
    return buf.getvalue()


def image_to_text(client: OpenAI, image_bytes: bytes, model: str = "gpt-4o-mini") -> str:
    """Image -> detailed product description via OpenAI vision."""
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {
                        "type": "text",
                        "text": (
                            "Describe this product in detail, including category, color, material, "
                            "style, key features, and typical use cases."
                        ),
                    },
                ],
            }
        ],
    )
    return resp.choices[0].message.content


def search_products(query: str, vectorstore: FAISS, top_k: int = 3) -> str:
    """Retrieve similar products (FAISS distance: lower = more similar)."""
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    if not results:
        return "No related products found."

    lines = []
    for i, (doc, score) in enumerate(results, start=1):
        lines.append(f"{i}. [Distance: {score:.4f}] {doc.page_content}")
    return "\n".join(lines)


# -------------------------
# Cached resources
# -------------------------
@st.cache_resource
def build_vectorstore(openai_api_key: str) -> FAISS:
    """Build once per session (or when key changes)."""
    docs = [Document(page_content=t, metadata={"product_id": i}) for i, t in enumerate(PRODUCT_DATABASE)]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
    return FAISS.from_documents(docs, embeddings)


# -------------------------
# Main Streamlit app
# -------------------------
def main():
    st.set_page_config(page_title="Product Search", layout="centered")
    st.title("Product Image Recognition and Retrieval (LangChain + OpenAI)")

    # Load env inside main (Streamlit-friendly)
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Sidebar config / diagnostics
    st.sidebar.header("Configuration")
    st.sidebar.write("OPENAI_API_KEY loaded:", bool(openai_api_key))

    if not openai_api_key:
        st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart Streamlit.")
        st.stop()

    # Build cached vector store
    vectorstore = build_vectorstore(openai_api_key)
    client = get_openai_client(openai_api_key)

    # UI
    st.write("### Upload a Product Image")
    uploaded_file = st.file_uploader("Choose a product image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Product Image", width=320)
    image_bytes = pil_to_bytes(img)

    # Action
    if st.button("Analyze Product"):
        try:
            with st.spinner("Analyzing image..."):
                description = image_to_text(client, image_bytes)

            st.subheader("Image Description")
            st.write(description)

            with st.spinner("Searching for similar products..."):
                results_text = search_products(description, vectorstore, top_k=3)

            st.subheader("Retrieval Results")
            st.text(results_text)

        except Exception as e:
            st.error("Error occurred while running the pipeline:")
            st.exception(e)

    with st.expander("View Product Database"):
        for i, p in enumerate(PRODUCT_DATABASE, start=1):
            st.write(f"{i}. {p}")


if __name__ == "__main__":
    main()
