import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from rag.rag_chain import RAGChain

# DummyRunnable to mock langchain Runnable behavior
class DummyRunnable:
    def __init__(self, func=None):
        self.func = func

    def __ror__(self, other):  # support dict | Runnable
        return self

    def __or__(self, other):  # support chaining runnables
        return self

    def invoke(self, arg):  # final invocation
        return "mocked answer"

@pytest.fixture
def dummy_config():
    return {
        "retriever_type": "basic",
        "vector_store": {"path": "/tmp/store", "collection_name": "test_col"},
        "llm": {"temperature": 0.0, "model": "test-model"}
    }

@patch('rag.rag_chain.get_retriever')
@patch('rag.rag_chain.ChatPromptTemplate.from_template')
@patch('rag.rag_chain.ChatOpenAI')
def test_init_retrieve_format(mock_chat_openai, mock_from_template, mock_get_retriever, dummy_config):
    # Mock retriever
    retriever = MagicMock()
    doc1 = Document(page_content="doc1")
    doc2 = Document(page_content="doc2")
    retriever.get_relevant_documents.return_value = [doc1, doc2]
    mock_get_retriever.return_value = retriever

    # Mock prompt and LLM
    mock_from_template.return_value = MagicMock()
    mock_chat_openai.return_value = MagicMock()

    # Initialize RAGChain
    chain = RAGChain(dummy_config)

    # Test retrieve_documents
    output = chain.retrieve_documents("query")
    assert output == "doc1\n\n" + "doc2"
    mock_get_retriever.assert_called_once_with(dummy_config)

    # Test format_docs
    formatted = chain.format_docs([doc1, doc2])
    assert formatted == "doc1\n\n" + "doc2"

@patch('rag.rag_chain.get_retriever')
@patch('rag.rag_chain.ChatPromptTemplate.from_template')
@patch('rag.rag_chain.ChatOpenAI')
@patch('rag.rag_chain.RunnableLambda')
@patch('rag.rag_chain.RunnablePassthrough')
@patch('rag.rag_chain.StrOutputParser')
def test_answer_question_flow(mock_str_parser, mock_passthrough, mock_lambda, mock_chat_openai, mock_from_template, mock_get_retriever, dummy_config):
    # Mock retriever
    retriever = MagicMock()
    doc = Document(page_content="only doc")
    retriever.get_relevant_documents.return_value = [doc]
    mock_get_retriever.return_value = retriever

    # Patch prompt, llm, and runnables to DummyRunnable
    mock_from_template.return_value = DummyRunnable()
    mock_chat_openai.return_value = DummyRunnable()
    mock_lambda.return_value = DummyRunnable()
    mock_passthrough.return_value = DummyRunnable()
    mock_str_parser.return_value = DummyRunnable()

    # Initialize and call answer_question
    chain = RAGChain(dummy_config)
    result = chain.answer_question("test question")
    assert result == "mocked answer"
    
    # Ensure pipeline components were created
    mock_lambda.assert_called_once()
    mock_passthrough.assert_called_once()
    mock_str_parser.assert_called_once()
