"""
Structured logging for UdaHub workflow.
Provides event logging for all agent decisions, tool usage, and routing choices.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum

class LogLevel(Enum):
    """Log event levels for workflow tracking."""
    CLASSIFICATION = "CLASSIFICATION"
    ROUTING = "ROUTING"
    TOOL_CALL = "TOOL_CALL"
    RESOLUTION = "RESOLUTION"
    ESCALATION = "ESCALATION"
    DEBUG = "DEBUG"

def setup_logging(filename: str = "udahub_workflow.log") -> logging.Logger:
    """Configure structured JSON logging for workflow events."""
    logger = logging.getLogger("udahub.workflow")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler with JSON formatting
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # JSON formatter for file
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "event_type": record.msg if isinstance(record.msg, str) else "UNKNOWN",
                "ticket_id": record.__dict__.get("ticket_id", "SYSTEM"),
                "message": record.getMessage(),
                "context": record.__dict__.get("context", {})
            }
            return json.dumps(log_entry)
    
    # Simple console formatter
    class ConsoleFormatter(logging.Formatter):
        def format(self, record):
            ticket_id = record.__dict__.get("ticket_id", "SYSTEM")
            event = record.msg if isinstance(record.msg, str) else "EVENT"
            msg = record.getMessage()
            return f"[{record.levelname}] {ticket_id} | {event}: {msg}"
    
    fh.setFormatter(JSONFormatter())
    ch.setFormatter(ConsoleFormatter())
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Global logger instance
logger = setup_logging("udahub_workflow.log")

def log_classification(ticket_id: str, classification: Dict[str, Any]):
    """Log ticket classification decision."""
    context = {
        "issue_type": classification.get("issue_type"),
        "urgency": classification.get("urgency"),
        "intent": classification.get("intent"),
        "confidence": classification.get("confidence"),
        "reasoning": classification.get("reasoning")
    }
    extra = {"ticket_id": ticket_id, "context": context}
    logger.info("Classification complete", extra=extra)

def log_routing_decision(ticket_id: str, route: str, classification: Dict[str, Any], 
                        confidence: float, threshold: float, reason: str):
    """Log routing decision."""
    context = {
        "route": route,
        "issue_type": classification.get("issue_type"),
        "confidence": confidence,
        "threshold": threshold,
        "reason": reason
    }
    extra = {"ticket_id": ticket_id, "context": context}
    logger.info(f"Routing decision: {route}", extra=extra)

def log_tool_call(ticket_id: str, tool_name: str, tool_input: Dict[str, Any]):
    """Log a tool function call."""
    context = {
        "tool": tool_name,
        "input_keys": list(tool_input.keys()) if isinstance(tool_input, dict) else []
    }
    extra = {"ticket_id": ticket_id, "context": context}
    logger.info(f"Tool called: {tool_name}", extra=extra)

def log_tool_result(ticket_id: str, tool_name: str, status: str, result_summary: str = ""):
    """Log a tool function result."""
    context = {
        "tool": tool_name,
        "status": status,
        "summary": result_summary
    }
    extra = {"ticket_id": ticket_id, "context": context}
    logger.info(f"Tool result: {tool_name} - {status}", extra=extra)

def log_resolution_attempt(ticket_id: str, issue_type: str):
    """Log start of resolution attempt."""
    context = {"issue_type": issue_type}
    extra = {"ticket_id": ticket_id, "context": context}
    logger.info("Resolution attempt starting", extra=extra)

def log_resolution_success(ticket_id: str, action_taken: str, confidence: float):
    """Log successful resolution."""
    context = {
        "action": action_taken,
        "confidence": confidence
    }
    extra = {"ticket_id": ticket_id, "context": context}
    logger.info(f"Resolution succeeded: {action_taken}", extra=extra)

def log_resolution_failed(ticket_id: str, reason: str):
    """Log failed resolution attempt."""
    context = {"reason": reason}
    extra = {"ticket_id": ticket_id, "context": context}
    logger.info(f"Resolution failed: {reason}", extra=extra)

def log_escalation(ticket_id: str, issue_type: str, urgency: str, reason: str):
    """Log ticket escalation."""
    context = {
        "issue_type": issue_type,
        "urgency": urgency,
        "reason": reason
    }
    extra = {"ticket_id": ticket_id, "context": context}
    logger.warning(f"Ticket escalated: {reason}", extra=extra)

def log_workflow_start(ticket_id: str, channel: str):
    """Log workflow start."""
    context = {"channel": channel}
    extra = {"ticket_id": ticket_id, "context": context}
    logger.info(f"Workflow started from {channel}", extra=extra)

def log_workflow_end(ticket_id: str, final_status: str):
    """Log workflow completion."""
    context = {"final_status": final_status}
    extra = {"ticket_id": ticket_id, "context": context}
    logger.info(f"Workflow completed: {final_status}", extra=extra)
