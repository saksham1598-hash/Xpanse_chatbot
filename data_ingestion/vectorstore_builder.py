import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent))
import json
from pathlib import Path
from uuid import uuid4
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from chunking_methods import chunk_text


from config import RAG_CONFIG, Chunk_param

# Set up logging
from utils.logger import get_logger
logger = get_logger()

def get_param_hash(params: dict) -> str:
    """Create a short hash for parameter dict for unique naming."""
    param_str = json.dumps(params, sort_keys=True)
    return hash(param_str) % (10 ** 8)

def create_documents(chunks, source_path, chunk_type="semantic"):
    """Wraps chunks as LangChain Documents with metadata."""
    if not chunks:
        logger.warning(f"No chunks generated for {source_path}")
        return []
        
    docs = []
    source_str = str(source_path)
    filename = os.path.basename(source_str)
    
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": source_str,
                "filename": filename,
                "chunk_type": chunk_type,
                "chunk_index": i
            }
        )
        docs.append(doc)
    return docs

def store_in_vector_db(docs, vectorstore_dir, collection_name):
    """Store documents in Chroma vector DB and persist."""
    if not docs:
        logger.warning("No documents to store in vector database")
        return
        
    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(vectorstore_dir),
    )

    uuids = [str(uuid4()) for _ in range(len(docs))]
    vectorstore.add_documents(documents=docs, ids=uuids)

    logger.info(f"Stored {len(docs)} documents in Chroma at: {vectorstore_dir}")

def process_all_markdown_files(strategy_name=None):
    """Processes all markdown files in the configured path using one or all strategies."""
    markdown_dir = RAG_CONFIG["paths"]["markdown_files"]
    files = list(markdown_dir.glob("*.md"))
    
    if not files:
        logger.warning(f"No markdown files found in {markdown_dir}")
        return

    strategies_to_run = [strategy_name] if strategy_name else list(Chunk_param.keys())

    for strategy in strategies_to_run:
        if strategy not in Chunk_param:
            logger.error(f"Unknown chunking strategy: {strategy}")
            continue
            
        config = Chunk_param[strategy]
        param_hash = get_param_hash(config["params"])
        vectorstore_name = f"{strategy}_{param_hash}"
        
        # Use correct path from config
        vectorstore_dir = RAG_CONFIG["vector_store"]["path"].parent / vectorstore_name
        vectorstore_dir.mkdir(parents=True, exist_ok=True)

        # Save params metadata
        with open(vectorstore_dir / "chunk_params.json", "w") as f:
            json.dump({
                "strategy": strategy,
                "function": config["function"],
                "params": config["params"]
            }, f, indent=4)

        all_docs = []
        logger.info(f" Processing markdown files with strategy: {strategy}")
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                chunks = chunk_text(text, strategy_name=strategy)
                
                # Handle case where chunk_text returns None due to an error
                if chunks is None:
                    logger.error(f"Failed to chunk file {file_path} with strategy {strategy}")
                    continue
                    
                documents = create_documents(chunks, file_path, chunk_type=strategy)
                all_docs.extend(documents)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

        if all_docs:
            store_in_vector_db(all_docs, vectorstore_dir, collection_name=vectorstore_name)
        else:
            logger.warning(f"No documents created for strategy {strategy}. Skipping vector store creation.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build vectorstores from markdown files using different chunking strategies.")
    parser.add_argument("--strategy", type=str, help="Optional: Run only one strategy (e.g., 'semantic_chunking')")

    args = parser.parse_args()
    process_all_markdown_files(strategy_name=args.strategy)