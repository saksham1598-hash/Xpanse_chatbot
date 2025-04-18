from config import RAG_CONFIG
from rag.rag_chain import RAGChain
from utils.logger import get_logger
import sys
from dotenv import load_dotenv
import os
load_dotenv()
logger = get_logger()

def main():
    logger.info("RAG application started.")

    if len(sys.argv) < 2:
        print("Usage: python app.py '<your question>'")
        logger.warning("No question provided as argument.")
        return

    question = sys.argv[1]
    logger.info(f"Received question: {question}")

    try:
        rag_pipeline = RAGChain(RAG_CONFIG)
        docs = rag_pipeline.retrieve_documents(question)
        logger.info(f"Retrieved {len(docs)} documents for the query.")

        answer = rag_pipeline.answer_question(question)
        logger.info("Generated answer successfully.")
        print("Answer:\n")
        print(answer)

    except Exception as e:
        logger.exception(f"Error occurred while processing the question: {e}")

if __name__ == "__main__":
    main()

