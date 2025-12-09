import os
from typing import Optional
from langchain_core.tools import Tool

# Check if Tavily is available
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


def create_web_search_tool(max_results: int = 5) -> Optional[Tool]:
    """
    Create a web search tool using Tavily API.
    
    Args:
        max_results: Maximum number of search results
        
    Returns:
        LangChain Tool for web search, or None if API key not configured
    """
    api_key = os.getenv("TAVILY_API_KEY")
    
    if not api_key or not TAVILY_AVAILABLE:
        return None
    
    client = TavilyClient(api_key=api_key)
    
    def search(query: str) -> str:
        """Search the web for current information."""
        try:
            response = client.search(
                query=query,
                max_results=max_results,
                include_answer=True,
                search_depth="basic"
            )
            
            # Format results with explicit citations
            results = []
            
            if response.get("answer"):
                results.append(f"[Web: Summary] {response['answer']}\n")
            
            for i, result in enumerate(response.get("results", []), 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                content = result.get("content", "")[:500]  # Truncate long content
                
                citation = f"[Web: {url or 'unknown'}]"
                results.append(
                    f"{citation} {title}\n{content}\n"
                )
            
            return "\n---\n".join(results) if results else "No results found."
            
        except Exception as e:
            return f"Web search error: {str(e)}"
    
    return Tool(
        name="web_search",
        description=(
            "Search the web for current information, research, and sources. "
            "Use this when you need up-to-date information, academic sources, "
            "or facts not in the uploaded documents. "
            "Input should be a search query."
        ),
        func=search
    )