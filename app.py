# =======================================================================
# app.py - Streamlit RAG Application 
# Uses the custom functions from functions.py:
#   load_and_chunk_document, create_embeddings, search_chunks, generate_answer
# =======================================================================

import streamlit as st
import tempfile
import os

from functions import (
    load_and_chunk_document,
    create_embeddings,
    search_chunks,
    generate_answer,
)

# -----------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------
st.set_page_config(page_title="Document Q&A with RAG", layout="wide")

st.title("📄 Document Q&A with RAG")
st.markdown("Upload a **PDF** or **TXT** file, retrieve relevant chunks, and get an AI-generated answer.")

# -----------------------------------------------------------------------
# SIDEBAR SETTINGS
# -----------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

model_options = ["command-r-08-2024", "command-a-03-2025"]
selected_model = st.sidebar.selectbox("Select Cohere Model", model_options, index=0)

chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=800, value=300, step=50)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=20, max_value=200, value=50, step=10)
top_k = st.sidebar.slider("Number of Chunks to Retrieve (k)", min_value=1, max_value=5, value=3)
temperature = st.sidebar.slider("Temperature (Creativity)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

st.sidebar.markdown("---")

# Cohere API key: prefer secrets, otherwise ask the user
try:
    cohere_api_key = st.secrets["COHERE_API_KEY"]
except Exception:
    cohere_api_key = st.sidebar.text_input(
        "Cohere API Key",
        type="password",
        help="Get a free trial key from https://dashboard.cohere.com/api-keys",
    )

st.sidebar.info("Upload a document, then click Search to retrieve chunks and generate an answer.")

# -----------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# -----------------------------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------------------------
st.subheader("1️⃣ Upload a Document")

uploaded_file = st.file_uploader("Choose a PDF or TXT file (max 10MB)", type=["pdf", "txt"])

if uploaded_file is not None:
    # Save uploaded file to a temp path so load_and_chunk_document can read it
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Only re-process if it's a new file or settings changed
    reprocess = st.session_state.doc_name != uploaded_file.name
    if st.button("🔄 Process Document") or reprocess:
        with st.spinner("Loading and chunking document..."):
            try:
                chunks = load_and_chunk_document(tmp_path, chunk_size=chunk_size, overlap=chunk_overlap)
                st.session_state.chunks = chunks
                st.session_state.doc_name = uploaded_file.name
                st.success(f"✅ Document split into {len(chunks)} chunks.")
            except Exception as e:
                st.error(f"Error processing document: {e}")
                st.stop()

        with st.spinner("Creating embeddings (this may take a moment)..."):
            try:
                embeddings = create_embeddings(st.session_state.chunks)
                st.session_state.embeddings = embeddings
                st.success(f"✅ Created {len(embeddings)} embeddings.")
            except Exception as e:
                st.error(f"Error creating embeddings: {e}")
                st.stop()

    os.unlink(tmp_path)

    if st.session_state.chunks:
        with st.expander("📚 Preview chunks"):
            for i, c in enumerate(st.session_state.chunks[:5]):
                st.text(f"Chunk {i+1}: {c[:200]}{'...' if len(c) > 200 else ''}")
            if len(st.session_state.chunks) > 5:
                st.caption(f"...and {len(st.session_state.chunks) - 5} more chunks.")

# -----------------------------------------------------------------------
# Q&A SECTION
# -----------------------------------------------------------------------
st.subheader("2️⃣ Ask a Question")

query = st.text_input("Enter your question about the document:")

search_clicked = st.button("🔍 Search")

if search_clicked:
    if not st.session_state.chunks or not st.session_state.embeddings:
        st.warning("Please upload and process a document first.")
    elif not query:
        st.warning("Please enter a question.")
    else:
        # 1. Search for relevant chunks
        with st.spinner("Searching for relevant chunks..."):
            top_chunks = search_chunks(
                query,
                st.session_state.chunks,
                st.session_state.embeddings,
                k=top_k,
            )

        # 2. Display retrieved chunks
        st.markdown("### 📖 Top Retrieved Chunks")
        for i, chunk in enumerate(top_chunks):
            with st.expander(f"Chunk {i + 1}"):
                st.write(chunk)

        # 3. Generate answer with Cohere (Task 5)
        st.markdown("### 💬 Generated Answer")
        if not cohere_api_key:
            st.warning("Enter your Cohere API key in the sidebar to generate an answer.")
        else:
            with st.spinner("Generating answer with Cohere..."):
                context = "\n\n---\n\n".join(top_chunks)
                answer = generate_answer(
                    query,
                    context,
                    cohere_api_key,
                    model_name=selected_model,
                    temperature=temperature,
                )
            st.success(answer)

st.divider()
st.caption("Built with Streamlit, Cohere API, and Sentence Transformers.")