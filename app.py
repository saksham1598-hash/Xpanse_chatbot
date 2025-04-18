from config import RAG_CONFIG
from rag.rag_chain import RAGChain
from utils.logger import get_logger
import sys
from dotenv import load_dotenv
import os

load_dotenv()


logger = get_logger()

def main():
    if len(sys.argv) < 2:
        print("Usage: python app.py '<your question>'")
        return

    question = sys.argv[1]
    rag_pipeline = RAGChain(RAG_CONFIG)

    docs = rag_pipeline.retrieve_documents(question)
    # print("=" * 100)
    # print("Retrieved Documents:\\n")
    # print(docs)
    # print("=" * 100)

    answer = rag_pipeline.answer_question(question)
    print("\\n" + "*" * 100)
    print("Answer:\\n")
    print(answer)
    # print("*" * 100)

if __name__ == "__main__":
    main()
