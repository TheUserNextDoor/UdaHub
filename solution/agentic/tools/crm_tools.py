from fastmcp import Client
from server.dependencies import MCP_SERVER_URL


async def lookup_customer(external_user_id: str) -> dict:
    """
    Calls crm/lookup_customer on the FastMCP server.

    Args:
        external_user_id: CultPass user_id (from udahub users.external_user_id)

    Returns:
        dict with user profile and subscription info
    """
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(
            "crm/lookup_customer",
            {"external_user_id": external_user_id},
        )
    return result


async def lookup_reservation(
    external_user_id: str,
    status_filter: str | None = None,
) -> dict:
    """
    Calls crm/lookup_reservation on the FastMCP server.

    Args:
        external_user_id: CultPass user_id
        status_filter:    Optional — 'confirmed', 'cancelled', 'pending'

    Returns:
        dict with list of reservations and experience details
    """
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(
            "crm/lookup_reservation",
            {
                "external_user_id": external_user_id,
                "status_filter":    status_filter,
            },
        )
    return result


async def issue_refund(
    external_user_id: str,
    reservation_id: str,
    reason: str,
) -> dict:
    """
    Calls crm/issue_refund on the FastMCP server.

    Args:
        external_user_id: CultPass user_id
        reservation_id:   Reservation to cancel and refund
        reason:           Reason for the refund

    Returns:
        dict with success status and updated reservation details
    """
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(
            "crm/issue_refund",
            {
                "external_user_id": external_user_id,
                "reservation_id":   reservation_id,
                "reason":           reason,
            },
        )
    return result
