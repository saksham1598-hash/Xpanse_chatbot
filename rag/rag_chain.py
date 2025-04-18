
from retriever.factory import get_retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from prompt import template
from langchain_core.runnables import RunnableLambda

class RAGChain:
    def __init__(self, config):
        self.retriever = get_retriever(config)
        self.prompt = ChatPromptTemplate.from_template(template)
        llm_config = config["llm"]
        self.llm = ChatOpenAI(temperature=llm_config["temperature"], model_name=llm_config["model"])

    def retrieve_documents(self, query):
        docs = self.retriever.get_relevant_documents(query) 
        print(docs)
        return "\n\n".join(doc.page_content for doc in docs)
        
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)


    def answer_question(self, query):
        rag_chain = ({
                "context": RunnableLambda(lambda q: self.format_docs(self.retriever.get_relevant_documents(q))),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain.invoke(query)
