# from typing import List, Tuple, Dict, Any
# from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# from rank_bm25 import BM25Okapi
# from utils.logger import get_logger

# logger = get_logger()

# def reciprocal_rank_fusion(
#     bm25_results: List[Tuple[Document, float]], 
#     semantic_results: List[Tuple[Document, float]], 
#     k: int = 40
# ) -> List[Tuple[Document, Dict[str, Any]]]:
#     """
#     Fuse two ranked lists of documents using Reciprocal Rank Fusion.
#     Returns documents with their fused scores and original scores for transparency.

#     Args:
#         bm25_results: List of (document, score) tuples from BM25 ranking
#         semantic_results: List of (document, score) tuples from semantic ranking
#         k: Constant in RRF formula (higher k gives less weight to high rankings)

#     Returns:
#         List of tuples containing (document, score_details)
#     """
#     fused_scores = {}
#     doc_map = {}
#     original_scores = {}

#     # Process BM25 results
#     for rank, (doc, score) in enumerate(bm25_results, start=1):
#         doc_id = id(doc)
#         fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank)
#         doc_map[doc_id] = doc
#         if doc_id not in original_scores:
#             original_scores[doc_id] = {"bm25_score": score, "semantic_score": None, "bm25_rank": rank}
#         else:
#             original_scores[doc_id]["bm25_score"] = score
#             original_scores[doc_id]["bm25_rank"] = rank

#     # Process semantic results
#     for rank, (doc, score) in enumerate(semantic_results, start=1):
#         doc_id = id(doc)
#         fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank)
#         doc_map[doc_id] = doc
#         if doc_id not in original_scores:
#             original_scores[doc_id] = {"bm25_score": None, "semantic_score": score, "semantic_rank": rank}
#         else:
#             original_scores[doc_id]["semantic_score"] = score
#             original_scores[doc_id]["semantic_rank"] = rank

#     # Create final ranked list with score details
#     ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
#     result = []
    
#     for doc_id, rrf_score in ranked:
#         doc = doc_map[doc_id]
#         score_details = {
#             "rrf_score": rrf_score,
#             **original_scores[doc_id]
#         }
#         result.append((doc, score_details))
    
#     logger.info("Reciprocal Rank Fusion produced %d ranked documents.", len(result))
#     return result

# # # --------------------------
# # # 1) Naive Retriever
# # # --------------------------


# class BasicRetriever:
#     """
#     A basic vector retriever that returns documents with their similarity scores.
#     """
#     def __init__(self, vector_db_path: str, collection_name: str, k: int = 5):
#         self.vector_db_path = vector_db_path
#         self.collection_name = collection_name
#         self.k = k
#         self.embeddings = OpenAIEmbeddings()
#         self.vectorstore = Chroma(
#             persist_directory=vector_db_path,
#             collection_name=collection_name,    
#             embedding_function=self.embeddings,
#         )
#         # Use search_type="similarity" to ensure we get cosine similarity scores
#         self.retriever = self.vectorstore.as_retriever(
#             search_kwargs={"k": k}, 
#             search_type="similarity"
#         )
#         logger.info("BasicRetriever instantiated with k=%d", k)

#     def get_relevant_documents(self, question: str) -> List[Document]:
#         """Standard LangChain interface - returns only documents without scores"""
#         # logger.info("BasicRetriever retrieving documents for: '%s'", question)
#         results = self.get_relevant_documents_with_scores(question)
#         return [doc for doc, _ in results]

#     def get_relevant_documents_with_scores(self, question: str) -> List[Tuple[Document, float]]:
#         """Enhanced method that returns documents with their similarity scores"""
#         logger.info("BasicRetriever retrieving documents with scores for: '%s'", question)
#         # Use the similarity_search_with_relevance_scores method to get scores
#         results = self.vectorstore.similarity_search_with_relevance_scores(question, k=self.k)
#         logger.info(f"Retrieved {len(results)} documents with scores")
#         return results
    
# # # --------------------------
# # # 2) BM25-reranker
# # # --------------------------


# class BM25RerankedRetriever:
#     """
#     Retriever that first gets semantic results and then reranks them using BM25.
#     """
#     def __init__(
#         self, 
#         vector_db_path: str, 
#         collection_name: str, 
#         semantic_k: int = 10, 
#         rerank_k: int = 5
#     ):
#         self.vector_db_path = vector_db_path
#         self.collection_name = collection_name
#         self.semantic_k = semantic_k  
#         self.rerank_k = rerank_k      
        
    
#         self.bm25 = None
        
#         self.embeddings = OpenAIEmbeddings()
#         self.vectorstore = Chroma(
#             persist_directory=vector_db_path,
#             collection_name=collection_name,
#             embedding_function=self.embeddings,
#         )
#         self.semantic_retriever = self.vectorstore.as_retriever(search_kwargs={"k": semantic_k})
#         logger.info("BM25RerankedRetriever instantiated: semantic_k=%d, rerank_k=%d", semantic_k, rerank_k)

#     def get_relevant_documents(self, question: str) -> List[Document]:
#         """Standard LangChain interface - returns only documents without scores"""
#         logger.info("BM25RerankedRetriever retrieving documents for: '%s'", question)
#         results = self.get_relevant_documents_with_scores(question)
#         return [doc for doc, _ in results]

#     def get_relevant_documents_with_scores(self, question: str) -> List[Tuple[Document, Dict[str, Any]]]:
#         """Enhanced method that returns documents with their semantic and BM25 scores"""
#         logger.info("BM25RerankedRetriever retrieving documents with scores for: '%s'", question)
        
#         # Get top n documents via semantic search
#         semantic_results = self.vectorstore.similarity_search_with_relevance_scores(
#             question, k=self.semantic_k
#         )
        
#         # Prepare corpus for BM25
#         corpus = [doc.page_content for doc, _ in semantic_results]
#         tokenized_corpus = [doc.lower().split() for doc in corpus]
        

#         self.bm25 = BM25Okapi(tokenized_corpus)
        
#         # Get BM25 scores for the query
#         query_tokens = question.lower().split()
#         bm25_scores = self.bm25.get_scores(query_tokens)
        

#         docs_with_scores = []
#         for i, (doc, semantic_score) in enumerate(semantic_results):
#             # Compute BM25 rank among these candidates.
#             # (Sort indices based on bm25_scores in descending order and find the position of i)
#             sorted_indices = sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)
#             bm25_rank = sorted_indices.index(i) + 1
            
#             docs_with_scores.append((
#                 doc, 
#                 {
#                     "semantic_score": semantic_score,
#                     "semantic_rank": i + 1,
#                     "bm25_score": bm25_scores[i],
#                     "bm25_rank": bm25_rank
#                 }
#             ))
        
#         # Sort the combined result based on BM25 score (or you can choose a different strategy)
#         reranked_docs = sorted(docs_with_scores, key=lambda x: x[1]["bm25_score"], reverse=True)

#         for idx, (doc, scores) in enumerate(reranked_docs, start=1):
#             snippet = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
#         logger.info("BM25RerankedRetriever returning top %d documents.", self.rerank_k)
#         return reranked_docs[:self.rerank_k]

# # # --------------------------
# # # 3) ReciprocalRankFusionRetriever
# # # --------------------------
# class ReciprocalRankFusionRetriever:
#     """
#     Retriever that combines semantic search and BM25 using Reciprocal Rank Fusion.
#     """
#     def __init__(
#         self, 
#         vector_db_path: str, 
#         collection_name: str, 
#         semantic_k: int = 10, 
#         bm25_k: int = 10, 
#         fusion_k: int = 5
#     ):
#         self.vector_db_path = vector_db_path
#         self.collection_name = collection_name
#         self.semantic_k = semantic_k  # Number for semantic retrieval
#         self.bm25_k = bm25_k          # Number for BM25 ranking
#         self.fusion_k = fusion_k      # Final number of documents to return
        

#         self.bm25 = None
        
#         self.embeddings = OpenAIEmbeddings()
#         self.vectorstore = Chroma(
#             persist_directory=vector_db_path,
#             collection_name=collection_name,
#             embedding_function=self.embeddings,
#         )
#         self.semantic_retriever = self.vectorstore.as_retriever(search_kwargs={"k": max(semantic_k, bm25_k)})
#         logger.info(
#             "ReciprocalRankFusionRetriever instantiated: semantic_k=%d, bm25_k=%d, fusion_k=%d",
#             semantic_k, bm25_k, fusion_k
#         )

#     def get_relevant_documents(self, question: str) -> List[Document]:
#         """Standard LangChain interface - returns only documents without scores"""
#         logger.info("ReciprocalRankFusionRetriever retrieving documents for: '%s'", question)
#         results = self.get_relevant_documents_with_scores(question)
#         return [doc for doc, _ in results]

#     def get_relevant_documents_with_scores(self, question: str) -> List[Tuple[Document, Dict[str, Any]]]:
#         """Enhanced method that returns documents with their fused scores and original scores"""
#         logger.info("ReciprocalRankFusionRetriever retrieving documents with scores for: '%s'", question)
        
#         # Get larger candidate set for both methods
#         candidate_size = max(self.semantic_k, self.bm25_k)
#         semantic_results = self.vectorstore.similarity_search_with_relevance_scores(
#             question, k=candidate_size
#         )
        
#         # Prepare corpus for BM25
#         corpus = [doc.page_content for doc, _ in semantic_results]
#         tokenized_corpus = [doc.lower().split() for doc in corpus]
    
#         self.bm25 = BM25Okapi(tokenized_corpus)
        
#         query_tokens = question.lower().split()
#         bm25_scores = self.bm25.get_scores(query_tokens)
        
  
#         bm25_results = []
#         for i, (doc, _) in enumerate(semantic_results):
#             bm25_results.append((doc, bm25_scores[i]))
        
#         bm25_sorted = sorted(bm25_results, key=lambda x: x[1], reverse=True)
#         bm25_top_k = bm25_sorted[:self.bm25_k]
#         semantic_top_k = semantic_results[:self.semantic_k]
    
#         fused_results = reciprocal_rank_fusion(bm25_top_k, semantic_top_k)
        
#         logger.info("ReciprocalRankFusionRetriever returning top %d documents.", self.fusion_k)
#         return fused_results[:self.fusion_k]

# retriever/methods.py
from typing import List, Tuple, Dict, Any, Optional, Union
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from utils.logger import get_logger
import time
from functools import wraps
from retriever.models import (
    SearchQuery, DocumentScore, RetrievalResult, RetrievalResultWithScores,
    BasicRetrieverConfig, BM25RerankedRetrieverConfig, ReciprocalRankFusionRetrieverConfig
)
from pydantic import ValidationError

logger = get_logger()

# Timeout decorator
def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Implement timeout logic here if needed
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

def reciprocal_rank_fusion(
    bm25_results: List[Tuple[Document, float]], 
    semantic_results: List[Tuple[Document, float]], 
    k: int = 40
) -> List[Tuple[Document, Dict[str, Any]]]:
    """
    Fuse two ranked lists of documents using Reciprocal Rank Fusion.
    Returns documents with their fused scores and original scores for transparency.

    Args:
        bm25_results: List of (document, score) tuples from BM25 ranking
        semantic_results: List of (document, score) tuples from semantic ranking
        k: Constant in RRF formula (higher k gives less weight to high rankings)

    Returns:
        List of tuples containing (document, score_details)
    """
    fused_scores = {}
    doc_map = {}
    original_scores = {}

    # Process BM25 results
    for rank, (doc, score) in enumerate(bm25_results, start=1):
        doc_id = id(doc)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank)
        doc_map[doc_id] = doc
        if doc_id not in original_scores:
            original_scores[doc_id] = {"bm25_score": score, "semantic_score": None, "bm25_rank": rank}
        else:
            original_scores[doc_id]["bm25_score"] = score
            original_scores[doc_id]["bm25_rank"] = rank

    # Process semantic results
    for rank, (doc, score) in enumerate(semantic_results, start=1):
        doc_id = id(doc)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank)
        doc_map[doc_id] = doc
        if doc_id not in original_scores:
            original_scores[doc_id] = {"bm25_score": None, "semantic_score": score, "semantic_rank": rank}
        else:
            original_scores[doc_id]["semantic_score"] = score
            original_scores[doc_id]["semantic_rank"] = rank

    # Create final ranked list with score details
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    
    for doc_id, rrf_score in ranked:
        doc = doc_map[doc_id]
        score_details = {
            "rrf_score": rrf_score,
            **original_scores[doc_id]
        }
        result.append((doc, score_details))
    
    logger.info("Reciprocal Rank Fusion produced %d ranked documents.", len(result))
    return result

# --------------------------
# 1) Basic Retriever
# --------------------------

class BasicRetriever:
    """
    A basic vector retriever that returns documents with their similarity scores.
    """
    def __init__(self, vector_db_path: str, collection_name: str, k: int = 5):
        try:
            # Validate config with Pydantic
            self.config = BasicRetrieverConfig(
                vector_db_path=vector_db_path,
                collection_name=collection_name,
                k=k
            )
            
            self.vector_db_path = self.config.vector_db_path
            self.collection_name = self.config.collection_name
            self.k = self.config.k
            
            self.embeddings = OpenAIEmbeddings()
            self.vectorstore = Chroma(
                persist_directory=self.vector_db_path,
                collection_name=self.collection_name,    
                embedding_function=self.embeddings,
            )
            # Use search_type="similarity" to ensure we get cosine similarity scores
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.k}, 
                search_type="similarity"
            )
            logger.info("BasicRetriever instantiated with k=%d", self.k)
            
        except ValidationError as e:
            logger.error(f"Validation error in BasicRetriever initialization: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing BasicRetriever: {e}")
            raise

    def _validate_query(self, question: str) -> SearchQuery:
        """Validate the input query"""
        try:
            return SearchQuery(query=question)
        except ValidationError as e:
            logger.error(f"Invalid query for BasicRetriever: {e}")
            raise ValueError(f"Invalid query: {e}")

    @timeout(30)
    def get_relevant_documents(self, question: str) -> List[Document]:
        """Standard LangChain interface - returns only documents without scores"""
        # Validate question
        validated_query = self._validate_query(question)
        
        start_time = time.time()
        logger.info("BasicRetriever retrieving documents for: '%s'", validated_query.query)
        
        try:
            results = self.get_relevant_documents_with_scores(validated_query.query)
            docs = [doc for doc, _ in results]
            
            # Create RetrievalResult model
            result = RetrievalResult(
                documents=docs,
                query=validated_query.query,
                timestamp=time.time(),
                retrieval_time=time.time() - start_time
            )
            
            return docs
        except Exception as e:
            logger.error(f"Error in BasicRetriever.get_relevant_documents: {e}")
            raise

    @timeout(30)
    def get_relevant_documents_with_scores(self, question: str) -> List[Tuple[Document, float]]:
        """Enhanced method that returns documents with their similarity scores"""
        # Validate question
        validated_query = self._validate_query(question)
        
        start_time = time.time()
        logger.info("BasicRetriever retrieving documents with scores for: '%s'", validated_query.query)
        
        try:
            # Use the similarity_search_with_relevance_scores method to get scores
            results = self.vectorstore.similarity_search_with_relevance_scores(validated_query.query, k=self.k)
            logger.info(f"Retrieved {len(results)} documents with scores")
            
            # Create RetrievalResultWithScores model
            result = RetrievalResultWithScores(
                documents=[doc for doc, _ in results],
                documents_with_scores=results,
                query=validated_query.query,
                timestamp=time.time(),
                retrieval_time=time.time() - start_time
            )
            
            return results
        except Exception as e:
            logger.error(f"Error in BasicRetriever.get_relevant_documents_with_scores: {e}")
            raise
    
# --------------------------
# 2) BM25-reranker
# --------------------------

class BM25RerankedRetriever:
    """
    Retriever that first gets semantic results and then reranks them using BM25.
    """
    def __init__(
        self, 
        vector_db_path: str, 
        collection_name: str, 
        semantic_k: int = 10, 
        rerank_k: int = 5
    ):
        try:
            # Validate config with Pydantic
            self.config = BM25RerankedRetrieverConfig(
                vector_db_path=vector_db_path,
                collection_name=collection_name,
                semantic_k=semantic_k,
                rerank_k=rerank_k
            )
            
            self.vector_db_path = self.config.vector_db_path
            self.collection_name = self.config.collection_name
            self.semantic_k = self.config.semantic_k
            self.rerank_k = self.config.rerank_k
            
            self.bm25 = None
            
            self.embeddings = OpenAIEmbeddings()
            self.vectorstore = Chroma(
                persist_directory=self.vector_db_path,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
            self.semantic_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.semantic_k})
            logger.info("BM25RerankedRetriever instantiated: semantic_k=%d, rerank_k=%d", 
                       self.semantic_k, self.rerank_k)
                       
        except ValidationError as e:
            logger.error(f"Validation error in BM25RerankedRetriever initialization: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing BM25RerankedRetriever: {e}")
            raise

    def _validate_query(self, question: str) -> SearchQuery:
        """Validate the input query"""
        try:
            return SearchQuery(query=question)
        except ValidationError as e:
            logger.error(f"Invalid query for BM25RerankedRetriever: {e}")
            raise ValueError(f"Invalid query: {e}")

    @timeout(30)
    def get_relevant_documents(self, question: str) -> List[Document]:
        """Standard LangChain interface - returns only documents without scores"""
        # Validate question
        validated_query = self._validate_query(question)
        
        start_time = time.time()
        logger.info("BM25RerankedRetriever retrieving documents for: '%s'", validated_query.query)
        
        try:
            results = self.get_relevant_documents_with_scores(validated_query.query)
            docs = [doc for doc, _ in results]
            
            # Create RetrievalResult model
            result = RetrievalResult(
                documents=docs,
                query=validated_query.query,
                timestamp=time.time(),
                retrieval_time=time.time() - start_time
            )
            
            return docs
        except Exception as e:
            logger.error(f"Error in BM25RerankedRetriever.get_relevant_documents: {e}")
            raise

    @timeout(30)
    def get_relevant_documents_with_scores(self, question: str) -> List[Tuple[Document, Dict[str, Any]]]:
        """Enhanced method that returns documents with their semantic and BM25 scores"""
        # Validate question
        validated_query = self._validate_query(question)
        
        start_time = time.time()
        logger.info("BM25RerankedRetriever retrieving documents with scores for: '%s'", validated_query.query)
        
        try:
            # Get top n documents via semantic search
            semantic_results = self.vectorstore.similarity_search_with_relevance_scores(
                validated_query.query, k=self.semantic_k
            )
            
            # Prepare corpus for BM25
            corpus = [doc.page_content for doc, _ in semantic_results]
            tokenized_corpus = [doc.lower().split() for doc in corpus]
            
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            # Get BM25 scores for the query
            query_tokens = validated_query.query.lower().split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            docs_with_scores = []
            for i, (doc, semantic_score) in enumerate(semantic_results):
                # Compute BM25 rank among these candidates
                sorted_indices = sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)
                bm25_rank = sorted_indices.index(i) + 1
                
                docs_with_scores.append((
                    doc, 
                    {
                        "semantic_score": semantic_score,
                        "semantic_rank": i + 1,
                        "bm25_score": bm25_scores[i],
                        "bm25_rank": bm25_rank
                    }
                ))
            
            # Sort the combined result based on BM25 score
            reranked_docs = sorted(docs_with_scores, key=lambda x: x[1]["bm25_score"], reverse=True)
            
            # Create RetrievalResultWithScores model
            result = RetrievalResultWithScores(
                documents=[doc for doc, _ in reranked_docs[:self.rerank_k]],
                documents_with_scores=reranked_docs[:self.rerank_k],
                query=validated_query.query,
                timestamp=time.time(),
                retrieval_time=time.time() - start_time
            )
            
            logger.info("BM25RerankedRetriever returning top %d documents.", self.rerank_k)
            return reranked_docs[:self.rerank_k]
        except Exception as e:
            logger.error(f"Error in BM25RerankedRetriever.get_relevant_documents_with_scores: {e}")
            raise

# --------------------------
# 3) ReciprocalRankFusionRetriever
# --------------------------
class ReciprocalRankFusionRetriever:
    """
    Retriever that combines semantic search and BM25 using Reciprocal Rank Fusion.
    """
    def __init__(
        self, 
        vector_db_path: str, 
        collection_name: str, 
        semantic_k: int = 10, 
        bm25_k: int = 10, 
        fusion_k: int = 5
    ):
        try:
            # Validate config with Pydantic
            self.config = ReciprocalRankFusionRetrieverConfig(
                vector_db_path=vector_db_path,
                collection_name=collection_name,
                semantic_k=semantic_k,
                bm25_k=bm25_k,
                fusion_k=fusion_k
            )
            
            self.vector_db_path = self.config.vector_db_path
            self.collection_name = self.config.collection_name
            self.semantic_k = self.config.semantic_k
            self.bm25_k = self.config.bm25_k
            self.fusion_k = self.config.fusion_k
            
            self.bm25 = None
            
            self.embeddings = OpenAIEmbeddings()
            self.vectorstore = Chroma(
                persist_directory=self.vector_db_path,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
            self.semantic_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": max(self.semantic_k, self.bm25_k)}
            )
            logger.info(
                "ReciprocalRankFusionRetriever instantiated: semantic_k=%d, bm25_k=%d, fusion_k=%d",
                self.semantic_k, self.bm25_k, self.fusion_k
            )
            
        except ValidationError as e:
            logger.error(f"Validation error in ReciprocalRankFusionRetriever initialization: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing ReciprocalRankFusionRetriever: {e}")
            raise

    def _validate_query(self, question: str) -> SearchQuery:
        """Validate the input query"""
        try:
            return SearchQuery(query=question)
        except ValidationError as e:
            logger.error(f"Invalid query for ReciprocalRankFusionRetriever: {e}")
            raise ValueError(f"Invalid query: {e}")

    @timeout(30)
    def get_relevant_documents(self, question: str) -> List[Document]:
        """Standard LangChain interface - returns only documents without scores"""
        # Validate question
        validated_query = self._validate_query(question)
        
        start_time = time.time()
        logger.info("ReciprocalRankFusionRetriever retrieving documents for: '%s'", validated_query.query)
        
        try:
            results = self.get_relevant_documents_with_scores(validated_query.query)
            docs = [doc for doc, _ in results]
            
            # Create RetrievalResult model
            result = RetrievalResult(
                documents=docs,
                query=validated_query.query,
                timestamp=time.time(),
                retrieval_time=time.time() - start_time
            )
            
            return docs
        except Exception as e:
            logger.error(f"Error in ReciprocalRankFusionRetriever.get_relevant_documents: {e}")
            raise

    @timeout(30)
    def get_relevant_documents_with_scores(self, question: str) -> List[Tuple[Document, Dict[str, Any]]]:
        """Enhanced method that returns documents with their fused scores and original scores"""
        # Validate question
        validated_query = self._validate_query(question)
        
        start_time = time.time()
        logger.info("ReciprocalRankFusionRetriever retrieving documents with scores for: '%s'", validated_query.query)
        
        try:
            # Get larger candidate set for both methods
            candidate_size = max(self.semantic_k, self.bm25_k)
            semantic_results = self.vectorstore.similarity_search_with_relevance_scores(
                validated_query.query, k=candidate_size
            )
            
            # Prepare corpus for BM25
            corpus = [doc.page_content for doc, _ in semantic_results]
            tokenized_corpus = [doc.lower().split() for doc in corpus]
        
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            query_tokens = validated_query.query.lower().split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            bm25_results = []
            for i, (doc, _) in enumerate(semantic_results):
                bm25_results.append((doc, bm25_scores[i]))
            
            bm25_sorted = sorted(bm25_results, key=lambda x: x[1], reverse=True)
            bm25_top_k = bm25_sorted[:self.bm25_k]
            semantic_top_k = semantic_results[:self.semantic_k]
        
            fused_results = reciprocal_rank_fusion(bm25_top_k, semantic_top_k)
            
            # Create RetrievalResultWithScores model
            result = RetrievalResultWithScores(
                documents=[doc for doc, _ in fused_results[:self.fusion_k]],
                documents_with_scores=fused_results[:self.fusion_k],
                query=validated_query.query,
                timestamp=time.time(),
                retrieval_time=time.time() - start_time
            )
            
            logger.info("ReciprocalRankFusionRetriever returning top %d documents.", self.fusion_k)
            return fused_results[:self.fusion_k]
        except Exception as e:
            logger.error(f"Error in ReciprocalRankFusionRetriever.get_relevant_documents_with_scores: {e}")
            raise


