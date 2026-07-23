# RAG Assignment — Document Q&A with Retrieval-Augmented Generation

A Streamlit app that lets you upload a PDF or TXT document, retrieves the most
relevant chunks for a question using local embeddings + cosine similarity, and
generates a grounded answer using Cohere's Chat API.

## Features

- **Custom document loader** — reads TXT/PDF and splits into overlapping chunks using `RecursiveCharacterTextSplitter`
- **Local embeddings** — uses `sentence-transformers` (`all-MiniLM-L6-v2`), no API cost for embedding
- **Semantic search** — cosine similarity via `sklearn` to retrieve the top-k most relevant chunks
- **Answer generation** — Cohere Chat API (`command-r-08-2024` / `command-a-03-2025`) generates an answer grounded strictly in the retrieved context
- **Configurable settings** — chunk size, overlap, top-k, temperature, and model all adjustable from the sidebar

## Project Structure

```
rag_assignment/
│
├── README.md
├── requirements.txt
├── app.py              # Complete Streamlit app
├── functions.py        # All custom functions (Tasks 1, 2, 3, 5)
├── notebook.ipynb       # Development/testing notebook
└── screenshots/         # Screenshots of the app running
```

## Functions (`functions.py`)

| Function | Purpose |
|---|---|
| `load_and_chunk_document(file_path, chunk_size=300, overlap=50)` | Reads a TXT/PDF file and splits it into chunks |
| `create_embeddings(chunks, model_name="all-MiniLM-L6-v2")` | Embeds chunks locally using SentenceTransformer |
| `search_chunks(query, chunks, embeddings, k=3)` | Returns top-k chunks most similar to the query via cosine similarity |
| `generate_answer(query, context, api_key, model_name, temperature)` | Calls Cohere Chat API to generate a grounded answer |

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get a free Cohere API key from [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys).

3. Either:
   - Create `.streamlit/secrets.toml` with:
     ```toml
     COHERE_API_KEY = "your-key-here"
     ```
   - Or just paste the key into the sidebar text input when the app runs.

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## How It Works

1. **Upload** a PDF or TXT file.
2. App calls `load_and_chunk_document()` to split it into overlapping text chunks.
3. App calls `create_embeddings()` to embed all chunks locally (no API cost).
4. You type a **question** and click **Search**.
5. App calls `search_chunks()` to find the top-k most relevant chunks via cosine similarity.
6. App calls `generate_answer()`, sending the retrieved chunks + your question to Cohere, which returns an answer grounded only in that context.

## Notebook

`notebook.ipynb` contains standalone test runs for each function:
- Task 1: chunking a sample document, showing chunk count and content
- Task 2: embedding the chunks, showing embedding shape
- Task 3: searching with a sample query, showing top-k results
- Task 5: generating an answer with Cohere using the retrieved context

## Tech Stack

- **Streamlit** — UI
- **LangChain** (`RecursiveCharacterTextSplitter`) — chunking
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) — local embeddings
- **scikit-learn** — cosine similarity search
- **Cohere** (`command-r-08-2024`) — answer generation
- **PyPDF2** — PDF text extraction

<img width="1920" height="1080" alt="Screenshot (377)" src="https://github.com/user-attachments/assets/74cfebdf-324a-41a3-8041-9d48c44d2637" />
<img width="1920" height="1080" alt="Screenshot (378)" src="https://github.com/user-attachments/assets/9fb17904-090e-462e-b146-6147f7ceba49" />
