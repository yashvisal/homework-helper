from typing import Optional
from langchain_core.tools import Tool
from src.vectorstore import VectorStore


def create_retrieval_tool(
    vector_store: VectorStore,
    source_filter: Optional[str] = None,
    k: int = 5,
    session_id: Optional[str] = None,
) -> Tool:
    """
    Create a LangChain tool for document retrieval.
    
    Args:
        vector_store: VectorStore instance to search
        source_filter: Optional filter to specific source
        k: Number of results to return
        session_id: Optional session ID to filter by
        
    Returns:
        LangChain Tool for retrieval
    """
    
    def retrieve(query: str) -> str:
        """Search uploaded documents for relevant information."""
        if vector_store is None:
            return "No relevant documents found."
        try:
            results = vector_store.search(
                query=query,
                k=k,
                source_filter=source_filter,
                session_id=session_id,
            )
        except Exception:
            return "No relevant documents found."
        
        if not results:
            return "No relevant documents found."
        
        # Format results with citation-friendly text
        formatted = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "Unknown")
            page = result["metadata"].get("page", "N/A")
            similarity = result.get("similarity", 0)
            text = result["text"]
            
            citation_tag = f"[Source: {source}, p.{page}]"
            formatted.append(
                f"{citation_tag}\n{text}\n(Relevance: {similarity:.2f})"
            )
        
        return "\n---\n".join(formatted)
    
    return Tool(
        name="retrieve_documents",
        description=(
            "Search through uploaded documents (PDFs, textbooks, notes) to find "
            "relevant information. Use this when you need to reference source material "
            "or find specific information from the user's documents. "
            "Input should be a search query describing what you're looking for."
        ),
        func=retrieve
    )

