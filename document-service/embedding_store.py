from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db"):
        """
        Initialize the embedding store.
        
        Args:
            model_name: The sentence-transformers model to use
            persist_directory: Where to store the ChromaDB data
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        
        # Create persistence directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.sentence_transformer = SentenceTransformer(model_name)
        
        # Set up ChromaDB with sentence transformers
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name="esg_documents",
            embedding_function=self.embedding_function,
            metadata={"description": "ESG document chunks"}
        )
        
        logger.info("Embedding store initialized successfully")
        
    def add_document_chunks(self, doc_id: str, chunks: List[str], metadata: Dict[str, Any]) -> List[str]:
        """
        Add document chunks to the collection.
        
        Args:
            doc_id: Unique identifier for the document
            chunks: List of text chunks
            metadata: Document metadata
            
        Returns:
            List of chunk IDs
        """
        if not chunks:
            logger.warning(f"No chunks to add for document {doc_id}")
            return []
            
        # Generate unique IDs for each chunk
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        
        # Prepare metadata for each chunk
        metadatas = []
        for i in range(len(chunks)):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            metadatas.append(chunk_metadata)
        
        # Add chunks to the collection
        logger.info(f"Adding {len(chunks)} chunks for document {doc_id}")
        self.collection.add(
            ids=chunk_ids,
            documents=chunks,
            metadatas=metadatas
        )
        
        return chunk_ids
        
    def search_documents(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query_text: The search query text
            n_results: Number of results to return
            
        Returns:
            List of search results with text and metadata
        """
        # Search the collection
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {},
                    "score": results["distances"][0][i] if results["distances"] and results["distances"][0] else None
                })
        
        return formatted_results
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.sentence_transformer.encode(text).tolist()
