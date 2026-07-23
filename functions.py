# =======================================================================
# functions.py
# Core RAG functions for the assignment:
#   1. load_and_chunk_document  -> Task 1
#   2. create_embeddings        -> Task 2
#   3. search_chunks             -> Task 3
#   4. generate_answer           -> Task 5
# =======================================================================

import os
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cohere


# ------------------------------------------------------------------
# TASK 1: Custom Document Loader
# ------------------------------------------------------------------
def load_and_chunk_document(file_path, chunk_size=300, overlap=50):
    """
    Loads text from a TXT or PDF file and splits it into chunks.

    Args:
        file_path (str): path to the .txt or .pdf file
        chunk_size (int): max characters per chunk
        overlap (int): number of overlapping characters between chunks

    Returns:
        list[str]: list of text chunks
    """
    # 1. Read the file based on its extension
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    if ext == ".pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .txt and .pdf are supported.")

    if not text.strip():
        raise ValueError("No text could be extracted from the file.")

    # 2. Split into chunks using RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)

    # 3. Return chunks
    return chunks


# ------------------------------------------------------------------
# TASK 2: Custom Embedding Function
# ------------------------------------------------------------------
def create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Creates embeddings for a list of text chunks using SentenceTransformer.

    Args:
        chunks (list[str]): list of text chunks
        model_name (str): sentence-transformers model to use

    Returns:
        list[list[float]]: list of embeddings, one per chunk
    """
    # 1. Load the model
    model = SentenceTransformer(model_name)

    # 2. Create embeddings
    embeddings = model.encode(chunks, show_progress_bar=False)

    # 3. Return embeddings as list of lists
    return [emb.tolist() for emb in embeddings]


# ------------------------------------------------------------------
# TASK 3: Search Function
# ------------------------------------------------------------------
def search_chunks(query, chunks, embeddings, k=3, model_name="all-MiniLM-L6-v2"):
    """
    Finds the top-k chunks most similar to the query using cosine similarity.

    Args:
        query (str): the user's question
        chunks (list[str]): list of text chunks (same order as embeddings)
        embeddings (list[list[float]]): precomputed chunk embeddings
        k (int): number of top results to return
        model_name (str): must match the model used in create_embeddings

    Returns:
        list[str]: top-k most similar chunks, ordered by similarity (highest first)
    """
    # 1. Embed the query (same model as chunks, so vectors are comparable)
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])

    # 2. Calculate cosine similarity between query and every chunk embedding
    embeddings_array = np.array(embeddings)
    similarities = cosine_similarity(query_embedding, embeddings_array)[0]

    # 3. Get top-k indices (highest similarity first)
    k = min(k, len(chunks))
    top_k_indices = np.argsort(similarities)[::-1][:k]

    # 4. Return top-k chunks
    return [chunks[i] for i in top_k_indices]


# ------------------------------------------------------------------
# TASK 5: Cohere Answer Generation
# ------------------------------------------------------------------
def generate_answer(query, context, api_key, model_name="command-r-08-2024", temperature=0.2):
    """
    Generates an answer to the query using Cohere's Chat API, grounded in context.

    Args:
        query (str): the user's question
        context (str): retrieved chunks joined together
        api_key (str): Cohere API key
        model_name (str): Cohere model to use
        temperature (float): creativity of the response

    Returns:
        str: the generated answer
    """
    # 1. Initialize Cohere client
    co_client = cohere.ClientV2(api_key=api_key)

    if not context:
        return "No relevant information found in the document."

    # 2. Create a prompt with context + query
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.
If the context does not contain the answer, say "I don't have enough information to answer."

Context:
{context}

Question:
{query}

Answer:"""

    # 3. Generate and return answer
    try:
        response = co_client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.message.content[0].text
    except Exception as e:
        return f"Error generating answer: {e}"