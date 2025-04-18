import logging
from pathlib import Path
import sys 
import os

from typing import List, Optional
from config import Chunk_param
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent))
# Set up logging


from utils.logger import get_logger
logger = get_logger()

def split_with_overlap(text: str, max_size: int, overlap: int) -> List[str]:
    sub_chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + max_size
        sub_chunk = text[start:end]
        sub_chunks.append(sub_chunk)
        if end >= text_length:
            break
        start = max(0, end - overlap)
    return sub_chunks


def semantic_chunk(text: str, model_name: str, max_chunk_size: int, overlap_size: int) -> Optional[List[str]]:
    try:
        logger.info("Using LangChain SemanticChunker with model: %s", model_name)

        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="standard_deviation"
        )

        documents = chunker.create_documents([text])
        final_chunks = []
        for doc in documents:
            chunk_text = doc.page_content
            if len(chunk_text) > max_chunk_size:
                logger.debug("Chunk too large (%d chars), applying overlap splitting.", len(chunk_text))
                sub_chunks = split_with_overlap(chunk_text, max_chunk_size, overlap_size)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk_text)

        logger.info("Total semantic chunks produced: %d", len(final_chunks))
        return final_chunks
    except Exception as e:
        logger.error("Error in semantic chunking: %s", str(e))
        return None


def sentence_token_chunk(text: str, model_name: str, chunk_token_limit: int) -> Optional[List[str]]:
    try:
        logger.info("Using custom sentence-based chunker with token limit: %d", chunk_token_limit)

        # Validate model exists
        model = SentenceTransformer(model_name)  # Only for init, not used here
        sentences = sent_tokenize(text)

        chunks = []
        current_chunk = []
        current_token_count = 0

        for sentence in sentences:
            token_count = len(sentence.split())
            if current_token_count + token_count > chunk_token_limit and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0
            current_chunk.append(sentence)
            current_token_count += token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        logger.info("Total sentence-based chunks produced: %d", len(chunks))
        return chunks
    except Exception as e:
        logger.error("Error in sentence token chunking: %s", str(e))
        return None


def recursive_chunk(text: str, chunk_size: int, chunk_overlap: int) -> Optional[List[str]]:
    try:
        logger.info("Using RecursiveCharacterTextSplitter: size=%d, overlap=%d", chunk_size, chunk_overlap)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_text(text)
        return chunks
    except Exception as e:
        logger.error("Error in recursive chunking: %s", str(e))
        return None


def chunk_text(text: str, strategy_name: str) -> Optional[List[str]]:
    if not text:
        logger.warning("Empty text provided for chunking")
        return []
        
    if strategy_name not in Chunk_param:
        logger.error(f"Unknown chunking strategy: {strategy_name}")
        return None

    strategy = Chunk_param[strategy_name]
    func_name = strategy["function"]
    params = strategy.get("params", {})

    logger.info("Applying chunking strategy: %s", strategy_name)
    logger.debug("Using function: %s with params: %s", func_name, params)

    try:
        if func_name == "sentence_token_chunk":
            return sentence_token_chunk(text, **params)
        elif func_name == "semantic_chunk":
            return semantic_chunk(text, **params)
        elif func_name == "recursive_chunk":
            return recursive_chunk(text, **params)
        else:
            logger.error(f"Chunking function '{func_name}' not implemented.")
            return None
    except Exception as e:
        logger.error("Error occurred during chunking: %s", str(e))
        return None