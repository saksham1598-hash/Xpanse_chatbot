# tests/test_retriever_models.py
import pytest
from retriever.models import (
    SearchQuery, 
    BasicRetrieverConfig,
    BM25RerankedRetrieverConfig,
    ReciprocalRankFusionRetrieverConfig
)

class TestSearchQuery:
    def test_valid_query(self):
        query = SearchQuery(query="test query")
        assert query.query == "test query"
        
    def test_query_too_short(self):
        with pytest.raises(ValueError, match="should have at least 2 characters"):
            SearchQuery(query="t")
            
    def test_query_whitespace(self):
        with pytest.raises(ValueError, match="Query cannot be empty"):
            SearchQuery(query="   ")

class TestBasicRetrieverConfig:
    def test_valid_config(self):
        config = BasicRetrieverConfig(
            vector_db_path="/path/to/db",
            collection_name="test_collection",
            k=10
        )
        assert config.vector_db_path == "/path/to/db"
        assert config.collection_name == "test_collection"
        assert config.k == 10
    
    def test_invalid_k(self):
        with pytest.raises(ValueError, match="should be less than or equal to 100"):
            BasicRetrieverConfig(
                vector_db_path="/path/to/db",
                collection_name="test_collection",
                k=101
            )

class TestBM25RerankedRetrieverConfig:
    def test_valid_config(self):
        config = BM25RerankedRetrieverConfig(
            vector_db_path="/path/to/db",
            collection_name="test_collection",
            semantic_k=20,
            rerank_k=10
        )
        assert config.vector_db_path == "/path/to/db"
        assert config.collection_name == "test_collection"
        assert config.semantic_k == 20
        assert config.rerank_k == 10
    
    def test_invalid_k_relation(self):
        with pytest.raises(ValueError, match="rerank_k must be less than or equal to semantic_k"):
            BM25RerankedRetrieverConfig(
                vector_db_path="/path/to/db",
                collection_name="test_collection",
                semantic_k=10,
                rerank_k=20
            )

class TestReciprocalRankFusionRetrieverConfig:
    def test_valid_config(self):
        config = ReciprocalRankFusionRetrieverConfig(
            vector_db_path="/path/to/db",
            collection_name="test_collection",
            semantic_k=20,
            bm25_k=20,
            fusion_k=10
        )
        assert config.vector_db_path == "/path/to/db"
        assert config.collection_name == "test_collection"
        assert config.semantic_k == 20
        assert config.bm25_k == 20
        assert config.fusion_k == 10
    
    def test_invalid_fusion_k(self):
        with pytest.raises(ValueError, match="fusion_k must be less than or equal to both semantic_k and bm25_k"):
            ReciprocalRankFusionRetrieverConfig(
                vector_db_path="/path/to/db",
                collection_name="test_collection",
                semantic_k=10,
                bm25_k=15,
                fusion_k=20
            )
