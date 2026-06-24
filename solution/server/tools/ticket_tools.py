import uuid
import json
from fastmcp import FastMCP
from server.dependencies import get_udahub_db
from data.models.udahub import Ticket, TicketMetadata, TicketMessage, RoleEnum, CustomerMemory

mcp = FastMCP("ticket-tools")


@mcp.tool()
def update_ticket_status(
    ticket_id: str,
    status: str,
    urgency: str | None = None,
) -> dict:
    """
    Updates the status of a ticket in ticket_metadata.
    Optionally appends an urgency tag (used on escalation path).

    Args:
        ticket_id: The ticket to update
        status:    New status — 'open', 'resolved', 'escalated', 'pending'
        urgency:   Optional urgency flag — 'low', 'medium', 'high', 'critical'

    Returns:
        dict confirming the update.
    """
    with get_udahub_db() as db:
        metadata = db.query(TicketMetadata).filter(
            TicketMetadata.ticket_id == ticket_id
        ).first()

        if not metadata:
            return {"error": f"Ticket metadata not found for '{ticket_id}'.", "success": False}

        metadata.status = status

        if urgency:
            existing_tags = metadata.tags or ""
            tag_list = [t.strip() for t in existing_tags.split(",") if t.strip()]
            urgency_tag = f"urgency:{urgency}"
            if urgency_tag not in tag_list:
                tag_list.append(urgency_tag)
            metadata.tags = ", ".join(tag_list)

        db.commit()

        return {
            "success":    True,
            "ticket_id":  ticket_id,
            "new_status": status,
        }


@mcp.tool()
def send_response(
    ticket_id: str,
    content: str,
    role: str = "agent",
) -> dict:
    """
    Writes a response message to ticket_messages — visible to the customer.

    Args:
        ticket_id: The ticket to respond to
        content:   The message content
        role:      Message role — 'agent' or 'ai' (default: 'agent')

    Returns:
        dict with the new message_id.
    """
    with get_udahub_db() as db:
        ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
        if not ticket:
            return {"error": f"Ticket '{ticket_id}' not found.", "success": False}

        try:
            role_enum = RoleEnum[role]
        except KeyError:
            return {"error": f"Invalid role '{role}'. Must be: {[r.value for r in RoleEnum]}", "success": False}

        message = TicketMessage(
            message_id=str(uuid.uuid4()),
            ticket_id=ticket_id,
            role=role_enum,
            content=content,
        )
        db.add(message)
        db.commit()
        db.refresh(message)

        return {
            "success":    True,
            "message_id": message.message_id,
            "ticket_id":  ticket_id,
            "role":       message.role.value,
        }


@mcp.tool()
def create_internal_note(
    ticket_id: str,
    content: str,
    role: str = "system",
) -> dict:
    """
    Writes an internal note to ticket_messages — visible to human agents only.

    Args:
        ticket_id: The ticket to annotate
        content:   The internal note content
        role:      Always 'system' for internal notes (default: 'system')

    Returns:
        dict with the new message_id.
    """
    with get_udahub_db() as db:
        ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
        if not ticket:
            return {"error": f"Ticket '{ticket_id}' not found.", "success": False}

        try:
            role_enum = RoleEnum[role]
        except KeyError:
            return {"error": f"Invalid role '{role}'.", "success": False}

        note = TicketMessage(
            message_id=str(uuid.uuid4()),
            ticket_id=ticket_id,
            role=role_enum,
            content=content,
        )
        db.add(note)
        db.commit()
        db.refresh(note)

        return {
            "success":    True,
            "message_id": note.message_id,
            "ticket_id":  ticket_id,
        }


@mcp.tool()
def update_long_term_memory(
    external_user_id: str,
    resolution_summary: str,
    issue_type: str,
    preferences: dict | None = None,
) -> dict:
    """
    Persists a resolution summary and preferences to long-term memory.
    Memory is keyed by external_user_id and survives across sessions.

    Args:
        external_user_id:   CultPass user_id — memory follows the customer
        resolution_summary: Brief summary of what was resolved
        issue_type:         The issue type that was resolved
        preferences:        Any new preferences learned (optional)

    Returns:
        dict confirming the memory update.
    """
    with get_udahub_db() as db:
        # Use default account for now
        account_id = "cultpass"
        
        memory = db.query(CustomerMemory).filter(
            CustomerMemory.account_id == account_id,
            CustomerMemory.external_user_id == external_user_id
        ).first()
        
        if not memory:
            memory = CustomerMemory(
                memory_id=str(uuid.uuid4()),
                account_id=account_id,
                external_user_id=external_user_id,
                past_resolutions="[]",
                past_issue_types="[]",
                preferences="{}"
            )
            db.add(memory)
        
        resolutions_json = str(memory.past_resolutions) if memory.past_resolutions is not None else "[]"
        issue_types_json = str(memory.past_issue_types) if memory.past_issue_types is not None else "[]"
        preferences_json = str(memory.preferences) if memory.preferences is not None else "{}"
        
        past_resolutions = json.loads(resolutions_json)
        past_issue_types = json.loads(issue_types_json)
        user_preferences = json.loads(preferences_json)
        
        past_resolutions.append(resolution_summary)
        
        if issue_type not in past_issue_types:
            past_issue_types.append(issue_type)
        
        if preferences:
            user_preferences.update(preferences)
        
        memory.past_resolutions = json.dumps(past_resolutions)
        memory.past_issue_types = json.dumps(past_issue_types)
        memory.preferences = json.dumps(user_preferences)
        
        db.commit()
        
        return {
            "success":          True,
            "external_user_id": external_user_id,
            "message":          f"Memory updated for user '{external_user_id}' and persisted to database.",
        }


@mcp.tool()
def get_long_term_memory(external_user_id: str) -> dict:
    """
    Retrieves long-term memory for a user.
    Called at graph entry to inject memory into the initial state.

    Args:
        external_user_id: CultPass user_id

    Returns:
        dict with past_resolutions, past_issue_types, and preferences.
    """
    with get_udahub_db() as db:
        account_id = "cultpass"
        
        memory = db.query(CustomerMemory).filter(
            CustomerMemory.account_id == account_id,
            CustomerMemory.external_user_id == external_user_id
        ).first()
        
        if not memory:
            return {
                "past_resolutions": [],
                "past_issue_types": [],
                "preferences": {},
            }
        
        return {
            "past_resolutions": json.loads(str(memory.past_resolutions) if memory.past_resolutions is not None else "[]"),
            "past_issue_types": json.loads(str(memory.past_issue_types) if memory.past_issue_types is not None else "[]"),
            "preferences": json.loads(str(memory.preferences) if memory.preferences is not None else "{}"),
        }
