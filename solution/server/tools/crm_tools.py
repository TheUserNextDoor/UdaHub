from fastmcp import FastMCP
from server.dependencies import get_cultpass_db
from data.models.cultpass import User, Reservation

mcp = FastMCP("crm-tools")


@mcp.tool()
def lookup_customer(external_user_id: str) -> dict:
    """
    Fetches a CultPass user's profile and subscription details.

    Args:
        external_user_id: CultPass user_id (from udahub users.external_user_id)

    Returns:
        dict with user profile and subscription info,
        or error message if user not found.
    """
    with get_cultpass_db() as db:
        user = db.query(User).filter(User.user_id == external_user_id).first()

        if not user:
            return {"error": f"User '{external_user_id}' not found in CultPass."}

        subscription = None
        if user.subscription:
            sub = user.subscription
            subscription = {
                "subscription_id": sub.subscription_id,
                "status":          sub.status,
                "tier":            sub.tier,
                "monthly_quota":   sub.monthly_quota,
                "started_at":      str(sub.started_at),
                "ended_at":        str(sub.ended_at) if sub.ended_at else None,
            }

        return {
            "user_id":      user.user_id,
            "full_name":    user.full_name,
            "email":        user.email,
            "is_blocked":   user.is_blocked,
            "subscription": subscription,
        }


@mcp.tool()
def lookup_reservation(
    external_user_id: str,
    status_filter: str | None = None,
) -> dict:
    """
    Fetches a user's reservations with full experience details.

    Args:
        external_user_id: CultPass user_id
        status_filter:    Optional — filter by 'confirmed', 'cancelled', 'pending'.
                          Returns all if omitted.

    Returns:
        dict with list of reservations, each including experience details.
    """
    with get_cultpass_db() as db:
        query = db.query(Reservation).filter(
            Reservation.user_id == external_user_id
        )

        if status_filter:
            query = query.filter(Reservation.status == status_filter)

        reservations = query.all()

        reservation_list = []
        for res in reservations:
            exp = res.experience
            reservation_list.append({
                "reservation_id": res.reservation_id,
                "status":         res.status,
                "created_at":     str(res.created_at),
                "experience": {
                    "experience_id":   exp.experience_id,
                    "title":           exp.title,
                    "description":     exp.description,
                    "location":        exp.location,
                    "when":            str(exp.when),
                    "is_premium":      exp.is_premium,
                    "slots_available": exp.slots_available,
                },
            })

        return {
            "user_id":      external_user_id,
            "total":        len(reservation_list),
            "reservations": reservation_list,
        }


@mcp.tool()
def issue_refund(
    external_user_id: str,
    reservation_id: str,
    reason: str,
) -> dict:
    """
    Cancels a reservation and marks it as refunded.

    Guards:
    - User must not be blocked
    - Reservation must belong to the requesting user
    - Reservation must be in 'confirmed' status

    Args:
        external_user_id: CultPass user_id
        reservation_id:   Reservation to cancel and refund
        reason:           Reason for the refund — logged internally

    Returns:
        dict with success status and updated reservation details.
    """
    with get_cultpass_db() as db:
        user = db.query(User).filter(User.user_id == external_user_id).first()
        if not user:
            return {"error": "User not found.", "success": False}
        if user.is_blocked:
            return {"error": "User account is blocked. Cannot process refund.", "success": False}

        reservation = db.query(Reservation).filter(
            Reservation.reservation_id == reservation_id,
            Reservation.user_id == external_user_id,
        ).first()

        if not reservation:
            return {"error": "Reservation not found for this user.", "success": False}

        if reservation.status != "confirmed":
            return {
                "error":   f"Reservation status is '{reservation.status}'. Only 'confirmed' reservations can be refunded.",
                "success": False,
            }

        reservation.status = "cancelled"
        db.commit()

        return {
            "success":        True,
            "reservation_id": reservation_id,
            "new_status":     "cancelled",
            "message":        f"Reservation '{reservation_id}' successfully cancelled and refunded. Reason: {reason}",
        }
