import os
from typing import List, Optional, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


class VectorStore:
    """
    ChromaDB-based vector store with OpenAI embeddings.
    
    Supports:
    - PDF document ingestion with automatic chunking
    - Semantic similarity search
    - Metadata filtering by source
    """
    
    # Chunking parameters
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    def __init__(
        self,
        collection_name: str = "homework_helper",
        persist_directory: str = "./chroma_data",
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        # Ensure persistence directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize OpenI embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            # Uses OPENAI_API_KEY from environment
        )
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def add_pdf(
        self,
        pdf_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """
        Add a PDF document to the vector store.
        
        Args:
            pdf_path: Path to the PDF file
            metadata: Optional additional metadata to attach
            session_id: Optional session ID for filtering
            
        Returns:
            Number of chunks added
        """
        # Load PDF
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        
        # Extract filename for metadata
        filename = Path(pdf_path).name
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # Include session_id in chunk_id to avoid collisions
            chunk_id = f"{session_id}_{filename}_{i}" if session_id else f"{filename}_{i}"
            ids.append(chunk_id)
            texts.append(chunk.page_content)
            
            # Build metadata
            chunk_metadata = {
                "source": filename,
                "page": chunk.metadata.get("page", 0),
                "chunk_index": i,
            }
            if session_id:
                chunk_metadata["session_id"] = session_id
            if metadata:
                chunk_metadata.update(metadata)
            metadatas.append(chunk_metadata)
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def add_text(
        self,
        text: str,
        source_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add raw text to the vector store.
        
        Args:
            text: Text content to add
            source_name: Name to identify this source
            metadata: Optional additional metadata
            
        Returns:
            Number of chunks added
        """
        # Create document and split
        doc = Document(page_content=text, metadata={"source": source_name})
        chunks = self.text_splitter.split_documents([doc])
        
        # Prepare data
        ids = []
        texts = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_name}_{i}"
            ids.append(chunk_id)
            texts.append(chunk.page_content)
            
            chunk_metadata = {
                "source": source_name,
                "chunk_index": i,
            }
            if metadata:
                chunk_metadata.update(metadata)
            metadatas.append(chunk_metadata)
        
        # Generate embeddings and add
        embeddings = self.embeddings.embed_documents(texts)
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def search(
        self,
        query: str,
        k: int = 5,
        source_filter: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for relevant chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            source_filter: Optional filter by source filename
            session_id: Optional session ID to filter by
            
        Returns:
            List of results with text, metadata, and similarity score
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Build where filter (combine source and session_id if both present)
        where_filter = None
        filters = []
        if source_filter:
            filters.append({"source": source_filter})
        if session_id:
            filters.append({"session_id": session_id})
        
        if len(filters) == 1:
            where_filter = filters[0]
        elif len(filters) > 1:
            where_filter = {"$and": filters}
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "similarity": 1 - results["distances"][0][i] if results["distances"] else 1
                })
        
        return formatted_results
    
    def get_sources(self) -> List[str]:
        """Get list of all unique source names in the store."""
        # Get all metadata
        all_data = self.collection.get(include=["metadatas"])
        sources = set()
        if all_data["metadatas"]:
            for meta in all_data["metadatas"]:
                if "source" in meta:
                    sources.add(meta["source"])
        return sorted(list(sources))
    
    def delete_source(self, source_name: str) -> int:
        """
        Delete all chunks from a specific source.
        
        Args:
            source_name: Source to delete
            
        Returns:
            Number of chunks deleted
        """
        # Get IDs for this source
        results = self.collection.get(
            where={"source": source_name},
            include=[]
        )
        
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0
    
    def clear(self):
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    @property
    def count(self) -> int:
        """Get total number of chunks in the store."""
        return self.collection.count()

