"""
Tavily web search tool for SemSI.
"""
from langchain_tavily import TavilySearch


def get_web_search_tool(max_results: int = 3) -> TavilySearch:
    """Create a Tavily web search tool instance."""
    return TavilySearch(max_results=max_results)
