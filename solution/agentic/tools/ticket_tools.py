from fastmcp import Client
from server.dependencies import MCP_SERVER_URL


async def update_ticket_status(
    ticket_id: str,
    status: str,
    urgency: str | None = None,
) -> dict:
    """
    Calls tickets/update_ticket_status on the FastMCP server.

    Args:
        ticket_id: The ticket to update
        status:    New status — 'open', 'resolved', 'escalated', 'pending'
        urgency:   Optional — 'low', 'medium', 'high', 'critical'

    Returns:
        dict confirming the update
    """
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(
            "tickets/update_ticket_status",
            {
                "ticket_id": ticket_id,
                "status":    status,
                "urgency":   urgency,
            },
        )
    return result


async def send_response(
    ticket_id: str,
    content: str,
    role: str = "agent",
) -> dict:
    """
    Calls tickets/send_response on the FastMCP server.

    Args:
        ticket_id: The ticket to respond to
        content:   The message content
        role:      'agent' or 'ai' (default: 'agent')

    Returns:
        dict with the new message_id
    """
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(
            "tickets/send_response",
            {
                "ticket_id": ticket_id,
                "content":   content,
                "role":      role,
            },
        )
    return result


async def create_internal_note(
    ticket_id: str,
    content: str,
    role: str = "system",
) -> dict:
    """
    Calls tickets/create_internal_note on the FastMCP server.

    Args:
        ticket_id: The ticket to annotate
        content:   Internal note content for human agents
        role:      Always 'system' (default: 'system')

    Returns:
        dict with the new message_id
    """
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(
            "tickets/create_internal_note",
            {
                "ticket_id": ticket_id,
                "content":   content,
                "role":      role,
            },
        )
    return result


async def update_long_term_memory(
    external_user_id: str,
    resolution_summary: str,
    issue_type: str,
    preferences: dict | None = None,
) -> dict:
    """
    Calls tickets/update_long_term_memory on the FastMCP server.

    Args:
        external_user_id:   CultPass user_id
        resolution_summary: Brief summary of what was resolved
        issue_type:         The issue type that was resolved
        preferences:        Any new preferences learned (optional)

    Returns:
        dict confirming the memory update
    """
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(
            "tickets/update_long_term_memory",
            {
                "external_user_id":   external_user_id,
                "resolution_summary": resolution_summary,
                "issue_type":         issue_type,
                "preferences":        preferences,
            },
        )
    return result


async def get_long_term_memory(external_user_id: str) -> dict:
    """
    Calls tickets/get_long_term_memory on the FastMCP server.
    Called at graph entry to inject memory into the initial state.

    Args:
        external_user_id: CultPass user_id

    Returns:
        dict with past_resolutions, past_issue_types, preferences
    """
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(
            "tickets/get_long_term_memory",
            {"external_user_id": external_user_id},
        )
    return result
