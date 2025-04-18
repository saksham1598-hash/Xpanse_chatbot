
import time
from retriever.factory import get_retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from prompt import template
from langchain_core.runnables import RunnableLambda
from utils.logger import get_logger

logger = get_logger()

class RAGChain:
    def __init__(self, config):
        self.retriever = get_retriever(config)
        logger.info("Retriever initialized.")

        self.prompt = ChatPromptTemplate.from_template(template)
        logger.info("Prompt template initialized.")

        llm_config = config["llm"]
        self.llm = ChatOpenAI(
            temperature=llm_config["temperature"],
            model_name=llm_config["model"]
        )
        logger.info(f"LLM initialized with model: {llm_config['model']}, temperature: {llm_config['temperature']}")

    def retrieve_documents(self, query):
        start_time = time.time()
        docs = self.retriever.get_relevant_documents(query)
        elapsed = time.time() - start_time
        logger.info(f"Retrieved {len(docs)} documents in {elapsed:.2f} seconds.")
        return "\n\n".join(doc.page_content for doc in docs)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def answer_question(self, query):
        try:
            rag_chain = ({
                "context": RunnableLambda(lambda q: self.format_docs(self.retriever.get_relevant_documents(q))),
                "question": RunnablePassthrough()
            } | self.prompt | self.llm | StrOutputParser())

            start_time = time.time()
            response = rag_chain.invoke(query)
            elapsed = time.time() - start_time

            logger.info(f"LLM response generated in {elapsed:.2f} seconds.")
            return response

        except Exception as e:
            logger.exception(f"Failed to generate answer for query: {query}")
            return "An error occurred while generating the answer."


