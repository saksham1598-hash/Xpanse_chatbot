# tests/test_retriever_factory.py
import pytest
from unittest.mock import patch, MagicMock
from retriever.factory import get_retriever, RetrieverFactoryConfig
from pydantic import ValidationError

# Test the RetrieverFactoryConfig validation
class TestRetrieverFactoryConfig:
    def test_valid_config(self):
        config = RetrieverFactoryConfig(
            retriever_type="basic",
            vector_store={"path": "/path/to/store", "collection_name": "test_collection"},
            retriever_params={"basic": {"k": 10}}
        )
        assert config.retriever_type == "basic"
        assert config.vector_store["path"] == "/path/to/store"
        assert config.vector_store["collection_name"] == "test_collection"
        assert config.retriever_params["basic"]["k"] == 10

    
    def test_invalid_retriever_type(self):
        with pytest.raises(ValidationError, match="should match pattern"):
            RetrieverFactoryConfig(
                retriever_type="unknown",
                vector_store={"path": "/path/to/store", "collection_name": "test_collection"}
            )
        

    def test_missing_vector_store_path(self):
        with pytest.raises(ValueError, match="Vector store must have a non-empty 'path'"):
            RetrieverFactoryConfig(
                retriever_type="basic",
                vector_store={"collection_name": "test_collection"}
            )

    def test_empty_vector_store_path(self):
        with pytest.raises(ValueError, match="Vector store must have a non-empty 'path'"):
            RetrieverFactoryConfig(
                retriever_type="basic",
                vector_store={"path": "", "collection_name": "test_collection"}
            )

    def test_missing_collection_name(self):
        with pytest.raises(ValueError, match="Vector store must have a non-empty 'collection_name'"):
            RetrieverFactoryConfig(
                retriever_type="basic",
                vector_store={"path": "/path/to/store"}
            )

    def test_empty_collection_name(self):
        with pytest.raises(ValueError, match="Vector store must have a non-empty 'collection_name'"):
            RetrieverFactoryConfig(
                retriever_type="basic",
                vector_store={"path": "/path/to/store", "collection_name": ""}
            )

# Test the retriever factory function
class TestGetRetriever:
    @patch('retriever.factory.BasicRetriever')
    def test_get_basic_retriever(self, mock_basic):
        # Setup the mock
        mock_instance = MagicMock()
        mock_basic.return_value = mock_instance

        # Call the factory with basic retriever config
        config = {
            "retriever_type": "basic",
            "vector_store": {"path": "/path/to/store", "collection_name": "test_collection"},
            "retriever_params": {"basic": {"k": 10}}
        }
        
        retriever = get_retriever(config)
        
        # Verify the correct retriever was created with right parameters
        mock_basic.assert_called_once_with(
            "/path/to/store",
            "test_collection",
            k=10
        )
        assert retriever == mock_instance

    @patch('retriever.factory.BM25RerankedRetriever')
    def test_get_bm25_retriever(self, mock_bm25):
        # Setup the mock
        mock_instance = MagicMock()
        mock_bm25.return_value = mock_instance

        # Call the factory with BM25 retriever config
        config = {
            "retriever_type": "bm25_rerank",
            "vector_store": {"path": "/path/to/store", "collection_name": "test_collection"},
            "retriever_params": {"bm25_rerank": {"semantic_k": 15, "rerank_k": 7}}
        }
        
        retriever = get_retriever(config)
        
        # Verify the correct retriever was created with right parameters
        mock_bm25.assert_called_once_with(
            "/path/to/store",
            "test_collection",
            semantic_k=15,
            rerank_k=7
        )
        assert retriever == mock_instance

    @patch('retriever.factory.ReciprocalRankFusionRetriever')
    def test_get_fusion_retriever(self, mock_fusion):
        # Setup the mock
        mock_instance = MagicMock()
        mock_fusion.return_value = mock_instance

        # Call the factory with fusion retriever config
        config = {
            "retriever_type": "fusion",
            "vector_store": {"path": "/path/to/store", "collection_name": "test_collection"},
            "retriever_params": {"fusion": {"semantic_k": 20, "bm25_k": 15, "fusion_k": 10}}
        }
        
        retriever = get_retriever(config)
        
        # Verify the correct retriever was created with right parameters
        mock_fusion.assert_called_once_with(
            "/path/to/store",
            "test_collection",
            semantic_k=20,
            bm25_k=15,
            fusion_k=10
        )
        assert retriever == mock_instance

    def test_invalid_config_structure(self):
        # Test with a completely invalid config
        with pytest.raises(ValueError):
            get_retriever({"invalid": "config"})

    @patch('retriever.factory.BasicRetriever')
    def test_default_parameters(self, mock_basic):
        # Setup the mock
        mock_instance = MagicMock()
        mock_basic.return_value = mock_instance

        # Call with minimal config and no parameters
        config = {
            "retriever_type": "basic",
            "vector_store": {"path": "/path/to/store", "collection_name": "test_collection"}
        }
        
        retriever = get_retriever(config)
        
        # Verify default parameters were used
        mock_basic.assert_called_once_with(
            "/path/to/store",
            "test_collection",
            k=5  # Default value
        )

    @patch('retriever.factory.logger')
    def test_exception_handling(self, mock_logger):
        # Set up a config that will cause an exception in the retriever initialization
        config = {
            "retriever_type": "basic",
            "vector_store": {"path": "/path/to/store", "collection_name": "test_collection"}
        }
        
        # Mock the BasicRetriever to raise an exception
        with patch('retriever.factory.BasicRetriever', side_effect=Exception("Test error")):
            # Verify the function raises the exception
            with pytest.raises(Exception, match="Test error"):
                get_retriever(config)
            
            # Verify the error was logged
            mock_logger.exception.assert_called_once()