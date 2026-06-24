from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from operator import add


class TicketInput(TypedDict):
    ticket_id: str                      
    account_id: str                     
    user_id: str                        
    external_user_id: str               
    channel: str                        
    status: str                         
    main_issue_type: Optional[str]     
    tags: Optional[str]                 
    latest_message: str                


class CustomerContext(TypedDict):
    full_name: str                      
    email: str                          
    is_blocked: bool                    
    subscription_status: Optional[str]  
    subscription_tier: Optional[str]    
    monthly_quota: Optional[int]        
    active_reservations: list[dict]     


class Classification(TypedDict):
    issue_type: str
    urgency: str
    intent: str
    confidence: float


class Resolution(TypedDict):
    action_taken: str          
    response_message: str       
    tool_calls_made: list[str]  
    resolved: bool             



class TicketState(TypedDict):
    ticket: TicketInput
    messages: Annotated[list[Any], add_messages]
    tool_usage: Annotated[list[str], add]

    classification: Optional[Classification]       
    customer_context: Optional[CustomerContext]     
    resolution: Optional[Resolution]               

    next_agent: Optional[str] 
    short_term_memory: dict
    long_term_memory: dict

    retrieved_context: Optional[list[str]]
    error: Optional[str]