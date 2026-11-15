"""LangChain StructuredTools with synchronous and asynchronous Store execution."""
import asyncio
from typing import Optional

from ._common import functions, optional_import_error


def tools(store, *, read_budget=None, include_remember=True):
    """Return LangChain tools bound to one Store; optional imports occur at this call."""
    remember, recall = functions(store, read_budget)
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise optional_import_error("langchain") from exc

    async def aremember(text: str, speaker: str = "user", time: str = "", request_id: Optional[str] = None) -> dict:
        return await asyncio.to_thread(remember, text, speaker, time, request_id)

    async def arecall(query: str) -> str:
        return await asyncio.to_thread(recall, query)

    result = []
    if include_remember:
        result.append(StructuredTool.from_function(remember, coroutine=aremember, name="muster_remember"))
    result.append(StructuredTool.from_function(recall, coroutine=arecall, name="muster_recall"))
    return result
