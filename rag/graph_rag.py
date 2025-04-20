# import asyncio
# from pathlib import Path
# import sys 
# import os
# sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent))
# from lightrag import LightRAG, QueryParam
# from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
# from lightrag.kg.shared_storage import initialize_pipeline_status

# # Import configuration
# from config import RAG_CONFIG
# from utils.logger import get_logger
# from dotenv import load_dotenv
# load_dotenv()
# logger = get_logger()


# # Load key parameters from configuration
# WORKING_DIR = str(RAG_CONFIG["graph_rag"]["working_dir"])
# MODE = RAG_CONFIG["graph_rag"]["mode"]
# MARKDOWN_DIR = Path(RAG_CONFIG["paths"]["markdown_files"])

# async def initialize_rag() -> LightRAG:
#     """Initializes the LightRAG instance and required storages."""
#     rag = LightRAG(
#         working_dir=WORKING_DIR,
#         embedding_func=openai_embed,
#         llm_model_func=gpt_4o_mini_complete
#     )
#     await rag.initialize_storages()
#     await initialize_pipeline_status()
#     return rag

# def read_markdown_file(file_path: Path) -> str:
#     """Reads the entire content of a markdown file."""
#     with file_path.open("r", encoding="utf-8") as file:
#         return file.read()

# def get_all_markdown_files(directory: Path):
#     """Returns a list of markdown files (*.md) from the given directory."""
#     return list(directory.glob("*.md"))

# def answer_question_KG(rag: LightRAG, query_text: str) -> str:
#     """Submits a query to the Grap RAG system using the provided system prompt."""
#     system_prompt = '''
# You are an expert escrow account assistant responding to user query about Data Sources provided below.

# ---Goal---

# Generate a concise response based on Data Sources and follow Response Rules. Data sources contain two parts: Knowledge Graph(KG) and Document Chunks(DC). Summarize all information in the provided Data Sources, and incorporate general knowledge relevant to the Data Sources. Do not include information not provided by Data Sources.
# When handling information with timestamps:
# 1. Each piece of information (both relationships and content) has a "created_at" timestamp indicating when we acquired this knowledge
# 2. When encountering conflicting information, consider both the content/relationship and the timestamp
# 3. Don't automatically prefer the most recent information - use judgment based on the context
# 4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

# ---Data Sources---

# 1. From Knowledge Graph(KG):
# {kg_context}

# 2. From Document Chunks(DC):
# {vector_context}

# ---Response Rules---

# - Target format and length: {response_type}
# - Use markdown formatting with appropriate section headings
# - Organize answer in sections focusing on main point or aspect of the answer
# - Use clear and descriptive section titles that reflect the content
# - List up to 4 most important reference sources at the end under "References" section. 
# - Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC)
# - **Include detailed information about the sources** - for example- entity name,source content( exact text from the source) or any information if available
# - If you don't know the answer, just say so. Do not make anything up.
# '''


#     return rag.query(
#         query_text,
#         param=QueryParam(mode=MODE),
#         system_prompt=system_prompt
#     )

# def main():
#     # Initialize RAG instance asynchronously.
#     rag = asyncio.run(initialize_rag())

#     # Retrieve and insert all markdown file contents into the RAG system.
#     markdown_files = get_all_markdown_files(MARKDOWN_DIR)
#     for file_path in markdown_files:
#         logger.info(f"Inserting: {file_path}")
#         content = read_markdown_file(file_path)
#         rag.insert(content)

#     # Define the query text.
#     query_text = (
#         "A borrower's initial escrow account statement shows an anticipated increase in property taxes in the second year of the loan. According to the regulation, what specific document is the originator or servicer encouraged to provide to the consumer in this situation, and what is the purpose of this disclosure?"
#     )

#     # Execute the query and print the response.
#     response = answer_question_KG(rag, query_text)
#     print("Query Response:\n", response)

# if __name__ == "__main__":
#     main()

import nest_asyncio
nest_asyncio.apply()

import os
import asyncio
from pathlib import Path
import sys
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent))
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

# Import configuration
from config import RAG_CONFIG
from utils.logger import get_logger
from dotenv import load_dotenv
load_dotenv()
logger = get_logger()

# Load key parameters from configuration
WORKING_DIR = str(RAG_CONFIG["graph_rag"]["working_dir"])
MODE = RAG_CONFIG["graph_rag"]["mode"]
MARKDOWN_DIR = Path(RAG_CONFIG["paths"]["markdown_files"])

# Global RAG instance to avoid reinitialization
_rag_instance = None

async def initialize_rag() -> LightRAG:
    """Initializes the LightRAG instance and required storages."""
    global _rag_instance
    
    # Return existing instance if already initialized
    if _rag_instance is not None:
        logger.info("Returning existing LightRAG instance")
        return _rag_instance
    
    logger.info("Initializing new LightRAG instance...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )
    
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    # Load documents
    logger.info("Loading and inserting documents...")
    await load_documents(rag)
    
    logger.info("LightRAG instance initialized and documents loaded")
    _rag_instance = rag
    return rag

def read_markdown_file(file_path: Path) -> str:
    """Reads the entire content of a markdown file."""
    with file_path.open("r", encoding="utf-8") as file:
        return file.read()

def get_all_markdown_files(directory: Path):
    """Returns a list of markdown files (*.md) from the given directory."""
    return list(directory.glob("*.md"))

async def load_documents(rag: LightRAG):
    """Load all markdown documents and insert them into the RAG system asynchronously."""
    markdown_files = get_all_markdown_files(MARKDOWN_DIR)
    logger.info(f"Found {len(markdown_files)} markdown files to insert")
    
    for file_path in markdown_files:
        logger.info(f"Inserting: {file_path}")
        content = read_markdown_file(file_path)
        # Use the async version of insert to avoid nested event loops
        await rag.ainsert(content)
    
    return len(markdown_files)

async def answer_question_KG_async(rag: LightRAG, query_text: str) -> str:
    """Async version of submitting a query to the Graph RAG system."""
    logger.info(f"Processing KG query: {query_text}")
    try:

        response = await rag.aquery(
            query_text,
            param=QueryParam(mode=MODE)
        )


        logger.info("KG query processed successfully")
        return response
    except Exception as e:
        logger.error(f"Error processing KG query: {e}")
        return f"Error processing knowledge graph query: {str(e)}"

# Function to be called from Streamlit
async def get_kg_answer(query_text):
    """Get a knowledge graph answer for the given query."""
    try:
        # Initialize RAG if needed
        rag = await initialize_rag()
        # Get answer
        return await answer_question_KG_async(rag, query_text)
    except Exception as e:
        logger.error(f"Error in get_kg_answer: {e}")
        return f"Error: {str(e)}"

# Non-async wrapper for testing outside of Streamlit
def answer_question_KG(rag, query_text):
    """Synchronous wrapper for answer_question_KG_async."""
    # This should only be used in contexts where asyncio.run() is safe
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If we're in an environment with a running loop (like Streamlit)
        raise RuntimeError("This function shouldn't be called from a running event loop. Use get_kg_answer instead.")
    else:
        # Only for testing outside of Streamlit
        return loop.run_until_complete(answer_question_KG_async(rag, query_text))

# For testing outside of Streamlit
# if __name__ == "__main__":
#     import nest_asyncio
#     nest_asyncio.apply()  # This allows nested asyncio loops
    
#     async def test_query():
#         rag = await initialize_rag()
#         query = "What are the main requirements for an escrow account?"
#         response = await answer_question_KG_async(rag, query)
#         print("Query Response:\n", response)
#         return response
    
#     # This is safe to run as a standalone script
#     asyncio.run(test_query())