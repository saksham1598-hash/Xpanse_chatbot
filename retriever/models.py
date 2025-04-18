from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Any, Optional, Union, Tuple
from langchain_core.documents import Document

class SearchQuery(BaseModel):
    """Validated search query"""
    query: str = Field(..., min_length=2, max_length=1000)
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()

class DocumentScore(BaseModel):
    """Document with score"""
    document: Any = Field(...)  # LangChain Document object
    score: float = Field(...)
    
    model_config = {
        "arbitrary_types_allowed": True  # To allow LangChain Document
    }

class RetrievalResult(BaseModel):
    """Result of a retrieval operation"""
    documents: List[Document] = Field(default_factory=list)
    query: str = Field(...)
    timestamp: float = Field(...)
    retrieval_time: float = Field(...)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class RetrievalResultWithScores(RetrievalResult):
    """Result of a retrieval operation with scores"""
    documents_with_scores: List[Tuple[Document, Any]] = Field(default_factory=list)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

class RetrieverConfig(BaseModel):
    """Base configuration for retrievers"""
    vector_db_path: str = Field(...)
    collection_name: str = Field(...)
    
    @field_validator('vector_db_path')
    @classmethod
    def validate_path(cls, v):
        if not v.strip():
            raise ValueError("Vector DB path cannot be empty")
        return v

class BasicRetrieverConfig(RetrieverConfig):
    """Configuration for BasicRetriever"""
    k: int = Field(default=5, ge=1, le=100)

class BM25RerankedRetrieverConfig(RetrieverConfig):
    """Configuration for BM25RerankedRetriever"""
    semantic_k: int = Field(default=10, ge=1, le=100)
    rerank_k: int = Field(default=5, ge=1, le=100)
    
    @model_validator(mode='after')
    def validate_k_values(self):
        if self.rerank_k > self.semantic_k:
            raise ValueError("rerank_k must be less than or equal to semantic_k")
        return self

class ReciprocalRankFusionRetrieverConfig(RetrieverConfig):
    """Configuration for ReciprocalRankFusionRetriever"""
    semantic_k: int = Field(default=10, ge=1, le=100)
    bm25_k: int = Field(default=10, ge=1, le=100)
    fusion_k: int = Field(default=5, ge=1, le=100)
    
    @model_validator(mode='after')
    def validate_k_values(self):
        if self.fusion_k > min(self.semantic_k, self.bm25_k):
            raise ValueError("fusion_k must be less than or equal to both semantic_k and bm25_k")
        return self