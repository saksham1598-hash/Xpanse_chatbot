# Xpanse Chatbot for Escrow Accounts

**Xpanse Chatbot** is a **Retrieval-Augmented Generation (RAG)**-based chatbot that enables users to query information about **Escrow Accounts** with high accuracy and reliability by leveraging robust RAG and Knowledge graphs. 

---

## 1. Data Ingestion

### 1.1 Knowledge Ingestion Pipeline
- Begins with web scraping from a seed URL:  
  e.g.- `https://www.consumerfinance.gov/rules-policy/regulations/1024/17/`
- Extracts and parses nested links in the parent webpage to generate knowledge base.
- Stores cleaned **Markdown** data with metadata for traceability, version control, and reproducibility.

---

### 1.2 Flexible Chunking Strategies
- Supports plug-and-play chunking strategies:
  - `semantic_chunking` using sentence embeddings.
  - `sentence_token_chunking` with sentence boundary detection and word count.
  - `recursive_chunking` for character-based chunking.
- All chunking behavior is **controlled via a centralized config**, including parameters like:
  - Chunk size
  - Overlap
  - Strategy-specific parameters

---

### 1.3 Vectorstore Builder
- Converts chunked markdown documents into vector embeddings using **OpenAI Embeddings**.
- Supports multiple chunking strategies and stores results in **Chroma** for efficient semantic search via config parameters.
    - All vectorstores are version-controlled via parameter-based hashing, ensuring reproducibility.
    - Attaches metadata (filename, chunk type, source) to each chunk for traceable retrieval.
- Fully configurable and CLI-ready, supporting easy integration into pipelines or automation flows.
    - CLI support to run a single strategy (`--strategy`) or all at once.

This module transforms ingested and chunked documents into vector embeddings for retrieval. It enables **rapid testing and prototyping** by switching strategies or parameters to create different vectorstores

---
## 2.  Retriever System

A modular retrieval layer that turns user queries into precise document hits, with three interchangeable strategies—fully controlled via the central config.

### 2.1  Retriever Factory 
  - **Pydantic‑validated** config (`retriever_type`, `retriever_params`, `vector_store`), enforcing correct parameters before instantiation.  
  - Initializes one of three retrievers based on `retriever_type`: `basic`, `bm25_rerank`, or `fusion`.  
  - Centralizes logging of type and params for full transparency.
---

### 2.2 Retriever Methods  
  - **BasicRetriever**  
    - Semantic search over Chroma + OpenAI embeddings.  
    - Returns top‑k documents by cosine similarity.  
  - **BM25RerankedRetriever**  
    - Two‑stage retrieval: first semantic (K = semantic_k), then lexical BM25 reranking (top rerank_k).  
    - Leverages `rank_bm25` on the semantic candidates for precision.  
  - **ReciprocalRankFusionRetriever**  
    - Combines semantic and BM25 lists via **Reciprocal Rank Fusion**.  
    - Configurable fusion constant (`fusion_k`) to balance contributions.  

- **Pydantic Validations**  
  - Validates non‑empty, trimmed queries (`min_length=2`, `max_length=1000`).  
  - Prevents whitespace‑only inputs with a custom validator, along with validations for other functions of the Retriever system

---
## 3.  RAG Module

This turns user queries into answers; first via standard RAG, then (optionally) via a knowledge‑graph fallback powered by LightRAG( https://arxiv.org/abs/2410.05779 ).

---

### 3.1  Standard RAG_Chain

- **Config‑driven Initialization**  
  - Instantiates the appropriate retriever via `get_retriever(config)` (basic, BM25‑rerank, or fusion).  
  - Loads prompt template (`ChatPromptTemplate`) and LLM (`ChatOpenAI`) using langchian and central config.
  - Measures and logs latency at each stage for monitoring.
---

### 3.2 Graph RAG (Knowledge Graph Fallback)

- **LightRAG Integration**  
  - Builds a persistent knowledge graph (KG) using `openai_embed` + `gpt_4o_mini_complete`  
  - loads all markdown docs into the KG for relationship and reference‑driven retrieval, which is powerful in this use case due to the complex relations and referential nature of the policies

  - **Standard RAG** → If answer quality is sufficient, return immediately.  
  - **KG RAG** → for deeper context or when “more detail” is requested, invokes Graph RAG.

- **Robust Fallback**  
  - Graph RAG often uncovers implicit links and hierarchies that pure vector retrieval may miss (see supplementary examples).

---

## 4.  App & Frontend

### Main Orchestration (`app.py`)
- **Entry Point**  
  - Parses a user question, logs receipt, and validates input.  
  - Instantiates `RAGChain` with all the central `RAG_CONFIG`.   
  - Prints the answer and logs success or exceptions.

---

### 4.1  Streamlit Frontend

- **Interactive Chat UI**  
  - Provides a text input for user queries and displays LLM answers in real time.  
  - Offers a “Use Knowledge Graph Fallback” toggle to invoke Graph RAG for answer generation in case Standard RAG falls short.

- **Feedback & Observability**  
  - Users can rate each response and leave comments.  
  - Feedback is captured and traced through **Langfuse**, enabling analysis of model performance and user satisfaction.

### 4.2 Langfuse Integration

- **End‑to‑End LLM Tracing**  
  Automatically logs every LLM call (prompt, response, latency) to Langfuse for fine‑grained analysis and debugging.
  
- **Rich Observability**  
  Captures metrics such as token usage, model errors, and throughput; dashboards provide real‑time insight into system health and performance.

- **Feedback Loop**  
  Records user ratings and comments alongside LLM traces, enabling data‑driven refinements to prompts, retrieval strategies, and model parameters.


## 5.  Installation & Usage

To run the Xpanse Chatbot:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/saksham1598-hash/Xpanse_chatbot.git
   cd Xpanse_chatbot
   ```

2. **Create & activate a Python environment**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # Linux / macOS
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**  
   Copy the .env and fill in the open-ai key (langfuse keys can be found in the supplementary folder):
 
   Edit `.env` to include:
   ```
   OPENAI_API_KEY=<sk-proj-->
   LANGFUSE_PUBLIC_KEY = <-langfuse-publick-key->
   LANGFUSE_SECRET_KEY=<-langfuse-private-key->
   OPENAI_API_BASE="https://api.openai.com/v1"
   LANGFUSE_HOST="http://localhost:3000"
   ```
5. **Start the Streamlit frontend**  
   ```bash
   streamlit run streamlit_frontend/app_streamlit.py
   ```
   Then open your browser at `http://localhost:8501` to:
   - "Enter queries" 
   - “Use Knowledge Graph”  
   - "Submit feedback (ratings & comments)" — all traced in Langfuse

6. **(Optional) Start Lanfuse to see LLM traces, interaction,  and feedback, etc**
    - Open a new terminal after starting docker dekstop

    ```
    docker-compose -f langfuse/docker-compose.yml up --build 
    ```

8. **(Optional) Run the CLI application**  
   ```bash
   python app.py "What is the latest regulation for escrow accounts?"
   ```
---

## 6. Key design factors that I aimed to achieve-

- **Modular Architecture & Plug‑and‑Play**  
  - Clear separation of ingestion, chunking, retrieval, and orchestration into dedicated modules, keeping in mind the  single responsibility principle 
  - Complete control over parameters via config file for maintainability and fast experimentation

- **Early Validation & Error Handling**  
  - Config‑driven parameters validated by Pydantic schemas (fail‑fast on type or schema mismatches)  

- **Traceability & Repeatability**  
  - Comprehensive metadata (timestamps, parameter hashes, etc) logged at every stage  
  - Version‑controlled artifacts (vectorstores, knowledge‑graph dumps, JSON configs)

- **Automation‑Ready & Maintainable**  
  - CLI commands, Streamlit entry points, and monitoring via Langfuse  

## Supplementary file - 
  **link** -https://drive.google.com/drive/folders/1EEHzuRGgdXri4qXXlX3k4OeGmAftqFzu?usp=sharing)
  - Bonus.md file contains answers to the bonus points - Moving this to AWS and the CI/CD production pipeline
  - Chatbot functionality and evolution of responses through different prompt engineering approaches, Standard RAG, and Knowledge Graph and fine-tuning approach for comparison
  - Langfuse details to view LLM tracing 






