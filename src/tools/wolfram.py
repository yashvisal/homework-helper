"""Wolfram Alpha tool for mathematical computations."""
import os
from typing import Optional
from langchain_core.tools import Tool

# Check if Wolfram Alpha is available
try:
    from langchain_community.utilities import WolframAlphaAPIWrapper
    WOLFRAM_AVAILABLE = True
except ImportError:
    WOLFRAM_AVAILABLE = False


def create_wolfram_tool() -> Optional[Tool]:
    """
    Create a Wolfram Alpha tool for mathematical computations.
    
    Requires WOLFRAM_ALPHA_APPID environment variable.
    
    Returns:
        LangChain Tool for Wolfram Alpha, or None if not configured
    """
    app_id = os.getenv("WOLFRAM_ALPHA_APPID")
    
    if not app_id:
        print("[WARN] WOLFRAM_ALPHA_APPID not set - Wolfram Alpha tool disabled")
        return None
    
    if not WOLFRAM_AVAILABLE:
        print("[WARN] langchain-community not installed - Wolfram Alpha tool disabled")
        return None
    
    try:
        wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=app_id)
        
        def compute(query: str) -> str:
            """Run a computation or query through Wolfram Alpha."""
            try:
                result = wolfram.run(query)
                if result:
                    return f"[Wolfram Alpha Result]\n{result}"
                return "No result from Wolfram Alpha."
            except Exception as e:
                return f"Wolfram Alpha error: {str(e)}"
        
        return Tool(
            name="wolfram_alpha",
            description=(
                "Use Wolfram Alpha for mathematical computations, equations, calculus, "
                "algebra, physics problems, unit conversions, and factual queries. "
                "Input should be a mathematical expression or question. "
                "Examples: 'solve x^2 + 2x - 8 = 0', 'derivative of sin(x)*cos(x)', "
                "'integrate x^2 from 0 to 5', '150 miles in kilometers'"
            ),
            func=compute
        )
    except Exception as e:
        print(f"[ERROR] Failed to create Wolfram Alpha tool: {e}")
        return None

