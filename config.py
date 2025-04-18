import os
from pathlib import Path

# === Base Directory Setup ===
# Updated to handle the new structure
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_INGESTION_DIR = BASE_DIR / "data_ingestion"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# === Chunking Parameters ===
Chunk_param = {
    "semantic_chunking": {
        "function": "semantic_chunk",
        "params": {
            "model_name": "all-MiniLM-L6-v2",
            "max_chunk_size": 3500,
            "overlap_size": 150
        }
    },
    "sentence_token_chunking": {
        "function": "sentence_token_chunk",
        "params": {
            "model_name": "all-MiniLM-L6-v2",
            "chunk_token_limit": 400
        }
    },
    "recursive_chunking": {
        "function": "recursive_chunk",
        "params": {
            "chunk_size": 1500,
            "chunk_overlap": 200
        }
    }
}

# === Vector Store Config ===
vector_store_path = DATA_DIR / "vector_stores" / "semantic_chunking_4421486" 
collection_name = vector_store_path.name

# === Main RAG Config ===
RAG_CONFIG = {
    "vector_store": {
        "type": "chroma",
        "path": vector_store_path,
        "collection_name": collection_name
    },

    "retriever_type": "bm25_rerank",  # Options: "basic", "bm25_rerank", or "fusion"

    "retriever_params": {
        "basic": {
            "k": 4
        },
        "bm25_rerank": {
            "semantic_k": 5,
            "rerank_k": 4
        },
        "fusion": {
            "semantic_k": 5,
            "bm25_k": 5,
            "fusion_k": 3
        }
    },

    # === LLM Config ===
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.0
    },

    # === Data Paths ===
    "paths": {
        "raw_data": DATA_DIR / "raw",
        "processed_data": DATA_DIR / "processed",
        "markdown_files": DATA_DIR / "markdown_files"
    }
}

# Ensure all main paths exist
for key in ["raw_data", "processed_data", "markdown_files"]:
    RAG_CONFIG["paths"][key].mkdir(parents=True, exist_ok=True)

# Add raw_links path separately
RAG_CONFIG["paths"]["raw_links"] = RAG_CONFIG["paths"]["raw_data"] / "links"
RAG_CONFIG["paths"]["raw_links"].mkdir(parents=True, exist_ok=True)

# === URL/WEB Scraping Config ===
RAG_CONFIG["URL_EXTRACT"] = {
    "save_markdown": True,  # Save markdown content extracted from each URL
    "markdown_files_path": RAG_CONFIG["paths"]["markdown_files"],  # Directory to save .md files
    "urls": [
        "https://www.consumerfinance.gov/rules-policy/regulations/1024/17/"
        # Add more URLs you want to scrape
    ],
    "raw_links_path": RAG_CONFIG["paths"]["raw_links"],  # Path to save extracted links
    "link_filters": {
        "exclude_regulations": ["17"],  # Regulation numbers to exclude(optional)
        "default_regulation": "1024"    # Default regulation number to add when missing(optional)
    }
}