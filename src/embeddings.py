"""Vector store implementation using FAISS and sentence-transformers."""
import logging
from typing import List, Optional, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

class FaissRetriever(BaseRetriever, BaseModel):
    """FAISS-based document retriever with proper Pydantic configuration"""
    
    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Required fields
    index: Any = Field(..., description="FAISS index instance")
    documents: List[Document] = Field(..., description="List of document contents")
    encoder: Any = Field(..., description="Sentence embedding model")

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents"""
        query_embedding = self.encoder.encode(query).astype('float32')
        _, indices = self.index.search(np.array([query_embedding]), k)
        return [self.documents[i] for i in indices[0] if i != -1]

class VectorStore:
    """Managed vector store with FAISS backend"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 encoder: Optional[SentenceTransformer] = None):
        self.encoder = encoder or SentenceTransformer(embedding_model)
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []

    def create_index(self, documents: List[Document]) -> FaissRetriever:
        """Create FAISS index from documents with validation"""
        if not documents:
            raise ValueError("Cannot create index from empty document list")

        logger.info(f"Creating index for {len(documents)} documents")
        
        document_texts = [doc.page_content for doc in documents]
        # Encode documents
        embeddings = self.encoder.encode(
            document_texts,
            convert_to_numpy=True
        ).astype('float32')

        # Create and populate index
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.documents = documents

        return FaissRetriever(
            index=self.index,
            documents=self.documents,
            encoder=self.encoder
        )