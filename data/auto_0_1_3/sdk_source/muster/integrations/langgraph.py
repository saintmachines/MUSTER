"""LangGraph tools and ToolNode, using MUSTER's append/search semantics."""
from ._common import optional_import_error


def tools(store, *, read_budget=None, include_remember=True):
    """Return LangChain-compatible tools for use in a LangGraph graph."""
    from .langchain import tools as make_tools

    return make_tools(store, read_budget=read_budget, include_remember=include_remember)


def tool_node(store, *, read_budget=None, include_remember=True):
    """Create a ToolNode that executes remember and recall against the supplied Store."""
    try:
        from langgraph.prebuilt import ToolNode
    except ImportError as exc:
        raise optional_import_error("langgraph") from exc
    return ToolNode(tools(store, read_budget=read_budget, include_remember=include_remember),
                    handle_tool_errors=False)
