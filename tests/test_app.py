import sys
import pytest
from unittest.mock import patch, MagicMock

# Import the main function from app.py
from app import main

@patch('app.RAGChain')
def test_usage_no_args(mock_ragchain, capsys):
    # Simulate running without arguments
    sys.argv = ['app.py']
    main()
    captured = capsys.readouterr()
    assert "Usage: python app.py" in captured.out
    mock_ragchain.assert_not_called()

@patch('app.RAGChain')
def test_main_with_question(mock_ragchain, capsys):
    # Simulate running with a question argument
    sys.argv = ['app.py', 'hello world']

    # Prepare a mocked RAGChain instance
    mocked_chain = MagicMock()
    mocked_chain.retrieve_documents.return_value = "doc1\n\ndoc2"
    mocked_chain.answer_question.return_value = "mocked answer"
    mock_ragchain.return_value = mocked_chain

    # Call main()
    main()

    # Capture output
    captured = capsys.readouterr()

    # Ensure retriever was called correctly
    mocked_chain.retrieve_documents.assert_called_once_with('hello world')
    mocked_chain.answer_question.assert_called_once_with('hello world')

    # Check that the printed output contains the answer section
    assert "Answer:" in captured.out
    assert "mocked answer" in captured.out
