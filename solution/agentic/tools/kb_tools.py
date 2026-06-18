"""
agentic/tools/kb_tools.py

MCP client wrapper for knowledge base tools.
Calls the FastMCP server's kb/* tools.

Used by: Resolver agent
"""

from fastmcp import Client
from server.dependencies import MCP_SERVER_URL


async def search_knowledge_base(
    query: str,
    account_id: str,
    top_k: int = 3,
) -> dict:
    """
    Calls kb/search_knowledge_base on the FastMCP server.

    Args:
        query:      The customer's message or search query
        account_id: Scopes search to this account's articles
        top_k:      Number of results to return (default 3)

    Returns:
        dict with 'results', 'query', 'total_found'
    """
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(
            "kb/search_knowledge_base",
            {
                "query":      query,
                "account_id": account_id,
                "top_k":      top_k,
            },
        )
    return result
