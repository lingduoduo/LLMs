# -----------------------------------------------
# Test model 
# -----------------------------------------------
from sentence_transformers import SentenceTransformer, util


model_name = 'paraphrase-MiniLM-L6-v2'

try:
    model = SentenceTransformer(model_name)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("Please check if your proxy settings are correct and active.")


# It’s recommended to load the model once during project initialization to avoid repeated loading.
def map_synonyms_by_similarity(main_terms: list, candidates: list, threshold: float = 0.6) -> dict:
    """
    Map candidate words to their closest standard terms by computing embedding similarity.

    Args:
        main_terms (list): List of standard terms.
        candidates (list): List of candidate synonym words.
        threshold (float): Similarity threshold for determining synonyms.

    Returns:
        dict: Mapping from standard terms to their matched synonyms.
    """
    _matched_synonyms = {term: [] for term in main_terms}

    if not main_terms or not candidates:
        return _matched_synonyms

    # Batch encode to improve efficiency
    embeddings = model.encode(main_terms + candidates, convert_to_tensor=True)
    term_embeddings = embeddings[:len(main_terms)]
    candidate_embeddings = embeddings[len(main_terms):]

    # Compute cosine similarity between standard terms and candidates
    similarity_matrix = util.cos_sim(term_embeddings, candidate_embeddings)

    for i, term in enumerate(main_terms):
        for j, candidate in enumerate(candidates):
            if similarity_matrix[i][j] > threshold:
                _matched_synonyms[term].append(candidate)

    return _matched_synonyms


# Example usage
main_terms_to_map = ["Convolutional Neural Network", "Neural Network"]
all_possible_synonyms = ["CNN", "ConvNet", "Artificial Neural Network", "NN", "Neural System", "Deep Learning Model"]

optimized_mapped_synonyms = map_synonyms_by_similarity(main_terms_to_map, all_possible_synonyms)
print("Example after optimization:", optimized_mapped_synonyms)


# -----------------------------------------------
#  Test Faiss
# -----------------------------------------------
import faiss
import numpy as np

def build_term_vector_index(term_glossary: dict, model: SentenceTransformer) -> tuple:
    """
    Build a FAISS index for term vectors.

    Args:
        term_glossary (dict): Mapping of standard terms to their synonyms.
        model (SentenceTransformer): Pre-loaded sentence transformer model.

    Returns:
        tuple: FAISS index containing term vectors.
    """
    terms_to_index = []

    # Iterate through the glossary and collect all standard terms and their synonyms
    for standard_term, info in term_glossary.items():
        terms_to_index.append(standard_term)
        if "synonyms" in info and isinstance(info["synonyms"], list):
            terms_to_index.extend(info["synonyms"])
    
    unique_terms_to_index = sorted(list(set(terms_to_index)))

    print("Building FAISS index for terms")
    embeddings = model.encode(unique_terms_to_index, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dimemsion = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimemsion)
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors of dimension {dimemsion}")
    return index, unique_terms_to_index

model = SentenceTransformer(model_name)

# 2. Prepare example term data
term_mapping_example = {
    "Convolutional Neural Network": {"synonyms": ["CNN", "ConvNet"]},
    "Transformer": {"synonyms": ["transformer", "TRANSFORMER"]},
    "Image Recognition": {"synonyms": ["image classification", "visual recognition"]}
}

# 3. Call the function to build the FAISS index
faiss_index, indexed_term_list = build_term_vector_index(term_mapping_example, model)

# 4. Display the results
print("\n--- FAISS index successfully built ---")
print(f"Number of vectors in FAISS index: {faiss_index.ntotal}")
print(f"List of indexed terms: {indexed_term_list}")

def search_similar_terms(query_text: str, model: SentenceTransformer, index: faiss.IndexFlatL2, term_list: list,  top_k: int = 5) -> list:
    """
    Search for terms similar to the query using the FAISS index.

    Args:
        query (str): The input query term.
        index (faiss.IndexFlatL2): The FAISS index containing term vectors.
        term_list (list): List of terms corresponding to the FAISS index.
        model (SentenceTransformer): Pre-loaded sentence transformer model.
        top_k (int): Number of top similar terms to retrieve.

    Returns:
        list: List of top_k similar terms.
    """
    print("\n--- Searching for similar terms ---")
    print(f"Query: {query_text}")

    # 1. Encode the query text
    query_vector = model.encode([query_text])
    query_vector = query_vector.astype("float32")

    # 2. Perform search in FAISS
    distances, indices = index.search(query_vector, k=3)

    # 3. Display results
    print("Search results:")
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        dist = distances[0][i]
        term = term_list[idx]
        print(f"{i+1}. {term} (distance: {dist:.4f})")
        results.append((term, dist))

    return results


search_similar_terms(query_text="CNN", model=model, index=faiss_index, term_list=indexed_term_list, top_k=3)