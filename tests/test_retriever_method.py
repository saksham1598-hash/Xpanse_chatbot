import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from retriever.methods import (
    reciprocal_rank_fusion,
    BasicRetriever,
    BM25RerankedRetriever,
    ReciprocalRankFusionRetriever,
    timeout
)
from retriever.models import (
    SearchQuery,
    BasicRetrieverConfig,
    BM25RerankedRetrieverConfig,
    ReciprocalRankFusionRetrieverConfig
)
from pydantic import ValidationError


class TestReciprocalRankFusion:
    def test_fusion_with_empty_lists(self):
        result = reciprocal_rank_fusion([], [])
        assert result == []
    
    def test_fusion_with_one_empty_list(self):
        doc = Document(page_content="test document")
        bm25_results = [(doc, 0.5)]
        result = reciprocal_rank_fusion(bm25_results, [])
        assert len(result) == 1
        assert result[0][0] == doc
        assert "rrf_score" in result[0][1]
        assert result[0][1]["bm25_score"] == 0.5
        assert result[0][1]["semantic_score"] is None
    
    def test_fusion_with_overlapping_documents(self):
        doc1 = Document(page_content="document 1")
        doc2 = Document(page_content="document 2")
        
        bm25_results = [(doc1, 0.8), (doc2, 0.6)]
        semantic_results = [(doc2, 0.9), (doc1, 0.7)]
        
        result = reciprocal_rank_fusion(bm25_results, semantic_results)
        
        # Check that we have 2 results
        assert len(result) == 2
        
        # Check the first result has the highest RRF score
        assert "rrf_score" in result[0][1]
        assert result[0][1]["rrf_score"] >= result[1][1]["rrf_score"]
        
        # Check that both documents have both scores
        for doc, scores in result:
            assert scores["bm25_score"] is not None
            assert scores["semantic_score"] is not None
    
    def test_fusion_with_custom_k(self):
        doc1 = Document(page_content="document 1")
        doc2 = Document(page_content="document 2")
        
        bm25_results = [(doc1, 0.8), (doc2, 0.6)]
        semantic_results = [(doc2, 0.9), (doc1, 0.7)]
        
        # With higher k, the scores should be closer together
        result_high_k = reciprocal_rank_fusion(bm25_results, semantic_results, k=100)
        
        # With lower k, the scores should be further apart
        result_low_k = reciprocal_rank_fusion(bm25_results, semantic_results, k=1)
        
        # The score difference should be greater with low k
        high_k_diff = result_high_k[0][1]["rrf_score"] - result_high_k[1][1]["rrf_score"]
        low_k_diff = result_low_k[0][1]["rrf_score"] - result_low_k[1][1]["rrf_score"]
        
        assert low_k_diff >= high_k_diff



@patch('retriever.methods.OpenAIEmbeddings')
@patch('retriever.methods.Chroma')
class TestBasicRetriever:
    def test_initialization(self, mock_chroma, mock_embeddings):
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        
        # Initialize retriever
        retriever = BasicRetriever(
            vector_db_path="/test/path",
            collection_name="test_collection",
            k=5
        )
        
        # Check if correct parameters were used
        mock_chroma.assert_called_once_with(
            persist_directory="/test/path",
            collection_name="test_collection",
            embedding_function=mock_embeddings_instance
        )
        
        assert retriever.k == 5
        assert retriever.vector_db_path == "/test/path"
        assert retriever.collection_name == "test_collection"
    
    def test_initialization_with_invalid_config(self, mock_chroma, mock_embeddings):
        # Test with invalid k
        with pytest.raises(ValidationError):
            BasicRetriever(
                vector_db_path="/test/path",
                collection_name="test_collection",
                k=101  # Over the maximum
            )
    
    def test_validate_query(self, mock_chroma, mock_embeddings):
        # Setup mock
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        
        retriever = BasicRetriever(
            vector_db_path="/test/path",
            collection_name="test_collection"
        )
        
        # Test valid query
        query = retriever._validate_query("test query")
        assert query.query == "test query"
        
        # Test invalid query
        with pytest.raises(ValueError):
            retriever._validate_query("")
    
    def test_get_relevant_documents(self, mock_chroma, mock_embeddings):
        # Setup mocks
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        
        retriever = BasicRetriever(
            vector_db_path="/test/path",
            collection_name="test_collection"
        )
        
        # Mock the get_relevant_documents_with_scores method
        doc1 = Document(page_content="test document 1")
        doc2 = Document(page_content="test document 2")
        retriever.get_relevant_documents_with_scores = MagicMock(
            return_value=[(doc1, 0.8), (doc2, 0.7)]
        )
        
        # Call the method
        result = retriever.get_relevant_documents("test query")
        
        # Check results
        assert len(result) == 2
        assert result[0] == doc1
        assert result[1] == doc2
        retriever.get_relevant_documents_with_scores.assert_called_once_with("test query")
    
    def test_get_relevant_documents_with_scores(self, mock_chroma, mock_embeddings):
        # Setup mocks
        doc1 = Document(page_content="test document 1")
        doc2 = Document(page_content="test document 2")
        
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
            (doc1, 0.8), (doc2, 0.7)
        ]
        mock_chroma.return_value = mock_vectorstore
        
        retriever = BasicRetriever(
            vector_db_path="/test/path",
            collection_name="test_collection"
        )
        
        # Call the method
        result = retriever.get_relevant_documents_with_scores("test query")
        
        # Check results
        assert len(result) == 2
        assert result[0][0] == doc1
        assert result[0][1] == 0.8
        assert result[1][0] == doc2
        assert result[1][1] == 0.7
        
        mock_vectorstore.similarity_search_with_relevance_scores.assert_called_once_with("test query", k=5)


@patch('retriever.methods.OpenAIEmbeddings')
@patch('retriever.methods.Chroma')
@patch('retriever.methods.BM25Okapi')
class TestBM25RerankedRetriever:
    def test_initialization(self, mock_bm25, mock_chroma, mock_embeddings):
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        
        # Initialize retriever
        retriever = BM25RerankedRetriever(
            vector_db_path="/test/path",
            collection_name="test_collection",
            semantic_k=10,
            rerank_k=5
        )
        
        # Check if correct parameters were used
        mock_chroma.assert_called_once_with(
            persist_directory="/test/path",
            collection_name="test_collection",
            embedding_function=mock_embeddings_instance
        )
        
        assert retriever.semantic_k == 10
        assert retriever.rerank_k == 5
        assert retriever.vector_db_path == "/test/path"
        assert retriever.collection_name == "test_collection"
    
    def test_initialization_with_invalid_config(self, mock_bm25, mock_chroma, mock_embeddings):
        # Test with invalid k relation
        with pytest.raises(ValidationError):
            BM25RerankedRetriever(
                vector_db_path="/test/path",
                collection_name="test_collection",
                semantic_k=5,
                rerank_k=10  # rerank_k > semantic_k
            )
    
    def test_get_relevant_documents(self, mock_bm25, mock_chroma, mock_embeddings):
        # Setup mocks
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        
        retriever = BM25RerankedRetriever(
            vector_db_path="/test/path",
            collection_name="test_collection"
        )
        
        # Mock the get_relevant_documents_with_scores method
        doc1 = Document(page_content="test document 1")
        doc2 = Document(page_content="test document 2")
        retriever.get_relevant_documents_with_scores = MagicMock(
            return_value=[(doc1, {"scores": "test"}), (doc2, {"scores": "test"})]
        )
        
        # Call the method
        result = retriever.get_relevant_documents("test query")
        
        # Check results
        assert len(result) == 2
        assert result[0] == doc1
        assert result[1] == doc2
        retriever.get_relevant_documents_with_scores.assert_called_once_with("test query")
    
    def test_get_relevant_documents_with_scores(self, mock_bm25, mock_chroma, mock_embeddings):
        # Setup mocks
        doc1 = Document(page_content="test document 1")
        doc2 = Document(page_content="test document 2")
        
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
            (doc1, 0.8), (doc2, 0.7)
        ]
        mock_chroma.return_value = mock_vectorstore
        
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.get_scores.return_value = [0.6, 0.9]
        mock_bm25.return_value = mock_bm25_instance
        
        retriever = BM25RerankedRetriever(
            vector_db_path="/test/path",
            collection_name="test_collection",
            semantic_k=2,
            rerank_k=2
        )
        
        # Call the method
        result = retriever.get_relevant_documents_with_scores("test query")
        
        # Check results
        assert len(result) == 2
        
        # The results should be sorted by BM25 score, so doc2 should be first
        assert result[0][0] == doc2
        assert result[0][1]["bm25_score"] == 0.9
        assert result[0][1]["semantic_score"] == 0.7
        
        assert result[1][0] == doc1
        assert result[1][1]["bm25_score"] == 0.6
        assert result[1][1]["semantic_score"] == 0.8
        
        # Check that BM25 was properly initialized
        mock_bm25.assert_called_once()
        mock_bm25_instance.get_scores.assert_called_once()


@patch('retriever.methods.OpenAIEmbeddings')
@patch('retriever.methods.Chroma')
@patch('retriever.methods.BM25Okapi')
@patch('retriever.methods.reciprocal_rank_fusion')
class TestReciprocalRankFusionRetriever:
    def test_initialization(self, mock_rrf, mock_bm25, mock_chroma, mock_embeddings):
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        
        # Initialize retriever
        retriever = ReciprocalRankFusionRetriever(
            vector_db_path="/test/path",
            collection_name="test_collection",
            semantic_k=10,
            bm25_k=10,
            fusion_k=5
        )
        
        # Check if correct parameters were used
        mock_chroma.assert_called_once_with(
            persist_directory="/test/path",
            collection_name="test_collection",
            embedding_function=mock_embeddings_instance
        )
        
        assert retriever.semantic_k == 10
        assert retriever.bm25_k == 10
        assert retriever.fusion_k == 5
        assert retriever.vector_db_path == "/test/path"
        assert retriever.collection_name == "test_collection"
    
    def test_initialization_with_invalid_config(self, mock_rrf, mock_bm25, mock_chroma, mock_embeddings):
        # Test with invalid fusion_k relation
        with pytest.raises(ValidationError):
            ReciprocalRankFusionRetriever(
                vector_db_path="/test/path",
                collection_name="test_collection",
                semantic_k=5,
                bm25_k=5,
                fusion_k=10  # fusion_k > semantic_k and bm25_k
            )
    
    def test_get_relevant_documents(self, mock_rrf, mock_bm25, mock_chroma, mock_embeddings):
        # Setup mocks
        mock_vectorstore = MagicMock()
        mock_chroma.return_value = mock_vectorstore
        
        retriever = ReciprocalRankFusionRetriever(
            vector_db_path="/test/path",
            collection_name="test_collection"
        )
        
        # Mock the get_relevant_documents_with_scores method
        doc1 = Document(page_content="test document 1")
        doc2 = Document(page_content="test document 2")
        retriever.get_relevant_documents_with_scores = MagicMock(
            return_value=[(doc1, {"scores": "test"}), (doc2, {"scores": "test"})]
        )
        
        # Call the method
        result = retriever.get_relevant_documents("test query")
        
        # Check results
        assert len(result) == 2
        assert result[0] == doc1
        assert result[1] == doc2
        retriever.get_relevant_documents_with_scores.assert_called_once_with("test query")
    
    def test_get_relevant_documents_with_scores(self, mock_rrf, mock_bm25, mock_chroma, mock_embeddings):
        # Setup mocks
        doc1 = Document(page_content="test document 1")
        doc2 = Document(page_content="test document 2")
        doc3 = Document(page_content="test document 3")
        
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
            (doc1, 0.8), (doc2, 0.7), (doc3, 0.6)
        ]
        mock_chroma.return_value = mock_vectorstore
        
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.get_scores.return_value = [0.6, 0.9, 0.5]
        mock_bm25.return_value = mock_bm25_instance
        
        # Setup RRF mock
        fused_results = [
            (doc2, {"rrf_score": 0.9, "bm25_score": 0.9, "semantic_score": 0.7}),
            (doc1, {"rrf_score": 0.8, "bm25_score": 0.6, "semantic_score": 0.8}),
            (doc3, {"rrf_score": 0.5, "bm25_score": 0.5, "semantic_score": 0.6})
        ]
        mock_rrf.return_value = fused_results
        
        retriever = ReciprocalRankFusionRetriever(
            vector_db_path="/test/path",
            collection_name="test_collection",
            semantic_k=2,
            bm25_k=2,
            fusion_k=2
        )
        
        # Call the method
        result = retriever.get_relevant_documents_with_scores("test query")
        
        # Check results
        assert len(result) == 2
        assert result[0][0] == doc2
        assert result[0][1]["rrf_score"] == 0.9
        
        assert result[1][0] == doc1
        assert result[1][1]["rrf_score"] == 0.8
        
        # Check that the RRF function was called correctly
        mock_rrf.assert_called_once()
        # Check fusion_k limit was applied
        assert len(result) == retriever.fusion_k


class TestTimeoutDecorator:
    def test_timeout_decorator(self):
        @timeout(5)
        def test_function():
            return "success"
        
        # The function should execute normally
        assert test_function() == "success"