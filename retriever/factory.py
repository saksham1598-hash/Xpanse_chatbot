# from retriever.methods import BasicRetriever, BM25RerankedRetriever, ReciprocalRankFusionRetriever
# from utils.logger import get_logger

# logger = get_logger()

# def get_retriever(config):
#     retriever_type = config["retriever_type"]
#     retriever_params = config["retriever_params"].get(retriever_type, {})
#     store = config["vector_store"]

#     logger.info(f"Initializing retriever of type: {retriever_type}")
#     logger.debug(f"Retriever parameters: {retriever_params}")
#     logger.debug(f"Vector store config: {store}")

#     try:
#         if retriever_type == "basic":
#             logger.info("Using BasicRetriever")
#             retriever = BasicRetriever(
#                 str(store["path"]),
#                 store["collection_name"],
#                 k=retriever_params.get("k", 5)
#             )

#         elif retriever_type == "bm25_rerank":
#             logger.info("Using BM25RerankedRetriever")
#             retriever = BM25RerankedRetriever(
#                 str(store["path"]),
#                 store["collection_name"],
#                 semantic_k=retriever_params.get("semantic_k", 10),
#                 rerank_k=retriever_params.get("rerank_k", 5)
#             )

#         elif retriever_type == "fusion":
#             logger.info("Using ReciprocalRankFusionRetriever")
#             retriever = ReciprocalRankFusionRetriever(
#                 str(store["path"]),
#                 store["collection_name"],
#                 semantic_k=retriever_params.get("semantic_k", 10),
#                 bm25_k=retriever_params.get("bm25_k", 10),
#                 fusion_k=retriever_params.get("fusion_k", 5)
#             )

#         else:
#             logger.error(f"Unknown retriever type: {retriever_type}")
#             raise ValueError(f"Unknown retriever type: {retriever_type}")

#         logger.info(f"{retriever_type} retriever initialized successfully.")
#         return retriever

#     except Exception as e:
#         logger.exception("Failed to initialize retriever.")
#         raise

# retriever/factory.py

from retriever.methods import BasicRetriever, BM25RerankedRetriever, ReciprocalRankFusionRetriever
from utils.logger import get_logger
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional

logger = get_logger()

class RetrieverFactoryConfig(BaseModel):
    """Configuration for retriever factory"""
    retriever_type: str = Field(..., pattern=r'^(basic|bm25_rerank|fusion)$')
    retriever_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    vector_store: Dict[str, Any] = Field(...)
    
    @field_validator('vector_store')
    @classmethod
    def validate_vector_store(cls, v):
        if 'path' not in v or not v['path']:
            raise ValueError("Vector store must have a non-empty 'path'")
        if 'collection_name' not in v or not v['collection_name']:
            raise ValueError("Vector store must have a non-empty 'collection_name'")
        return v

def get_retriever(config: Dict[str, Any]):
    try:
        # Validate the config first
        validated_config = RetrieverFactoryConfig(**config)
        
        retriever_type = validated_config.retriever_type
        retriever_params = validated_config.retriever_params.get(retriever_type, {})
        store = validated_config.vector_store

        logger.info(f"Initializing retriever of type: {retriever_type}")
        logger.debug(f"Retriever parameters: {retriever_params}")
        logger.debug(f"Vector store config: {store}")

        if retriever_type == "basic":
            logger.info("Using BasicRetriever")
            retriever = BasicRetriever(
                str(store["path"]),
                store["collection_name"],
                k=retriever_params.get("k", 5)
            )

        elif retriever_type == "bm25_rerank":
            logger.info("Using BM25RerankedRetriever")
            retriever = BM25RerankedRetriever(
                str(store["path"]),
                store["collection_name"],
                semantic_k=retriever_params.get("semantic_k", 10),
                rerank_k=retriever_params.get("rerank_k", 5)
            )

        elif retriever_type == "fusion":
            logger.info("Using ReciprocalRankFusionRetriever")
            retriever = ReciprocalRankFusionRetriever(
                str(store["path"]),
                store["collection_name"],
                semantic_k=retriever_params.get("semantic_k", 10),
                bm25_k=retriever_params.get("bm25_k", 10),
                fusion_k=retriever_params.get("fusion_k", 5)
            )

        logger.info(f"{retriever_type} retriever initialized successfully.")
        return retriever

    except Exception as e:
        logger.exception(f"Failed to initialize retriever: {e}")
        raise


