"""
Base class for privacy and security tools, adapted for LangChain.
"""
from langchain_core.tools import BaseTool as LangChainBaseTool
from pydantic import BaseModel


class SemSIBaseTool(LangChainBaseTool):
    """Base class for all SemSI privacy tools as LangChain tools."""
    pass
