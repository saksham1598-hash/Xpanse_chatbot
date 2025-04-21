import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
import os
import sys
from pathlib import Path
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent))

from src.rag.rag_chain import RAGChain

# DummyRunnable to mock langchain Runnable behavior
class DummyRunnable:
    def __init__(self, func=None):
        self.func = func

    def __ror__(self, other):  
        return self

    def __or__(self, other):  
        return self

    def invoke(self, arg, **kwargs):  # accept extra kwargs like config
        return "mocked answer"

@pytest.fixture
def dummy_config():
    return {
        "retriever_type": "basic",
        "vector_store": {"path": "/tmp/store", "collection_name": "test_col"},
        "llm": {"temperature": 0.0, "model": "test-model"}
    }

@patch('src.rag.rag_chain.get_retriever')
@patch('src.rag.rag_chain.ChatPromptTemplate.from_template')
@patch('src.rag.rag_chain.ChatOpenAI')
def test_init_retrieve_format(mock_chat_openai, mock_from_template, mock_get_retriever, dummy_config):
    # Mock retriever
    retriever = MagicMock()
    doc1 = Document(page_content="doc1")
    doc2 = Document(page_content="doc2")
    retriever.get_relevant_documents.return_value = [doc1, doc2]
    mock_get_retriever.return_value = retriever

    # Mock prompt and LLM runnables
    mock_from_template.return_value = DummyRunnable()
    mock_chat_openai.return_value = DummyRunnable()
    chain = RAGChain(dummy_config)

  
    mock_get_retriever.assert_called_once_with(dummy_config)

    # Test retrieve_documents returns list of Documents
    docs = chain.retrieve_documents("query")
    assert isinstance(docs, list)
    assert [d.page_content for d in docs] == ["doc1", "doc2"]

@patch('src.rag.rag_chain.langfuse_handler.get_trace_id')
@patch('src.rag.rag_chain.get_retriever')
@patch('src.rag.rag_chain.ChatPromptTemplate.from_template')
@patch('src.rag.rag_chain.ChatOpenAI')
@patch('src.rag.rag_chain.RunnableLambda')
@patch('src.rag.rag_chain.RunnablePassthrough')
@patch('src.rag.rag_chain.StrOutputParser')
def test_answer_question_flow(
    mock_str_parser,
    mock_passthrough,
    mock_lambda,
    mock_chat_openai,
    mock_from_template,
    mock_get_retriever,
    mock_get_trace_id,
    dummy_config
):
    # Mock retriever
    retriever = MagicMock()
    doc = Document(page_content="only doc")
    retriever.get_relevant_documents.return_value = [doc]
    mock_get_retriever.return_value = retriever

    mock_get_trace_id.return_value = "test-trace-id"

    # Patch prompt, LLM, and runnables to DummyRunnable
    mock_from_template.return_value = DummyRunnable()
    mock_chat_openai.return_value = DummyRunnable()
    mock_lambda.return_value = DummyRunnable()
    mock_passthrough.return_value = DummyRunnable()
    mock_str_parser.return_value = DummyRunnable()

  
    chain = RAGChain(dummy_config)
    result = chain.answer_question("test question")

    # Validate response and trace capture
    assert result == "mocked answer"
    assert chain.last_trace_id == "test-trace-id"
    mock_get_trace_id.assert_called_once()
