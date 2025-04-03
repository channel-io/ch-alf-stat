
from pydantic import BaseModel
from typing import Optional


class ALFFactCheck(BaseModel):
    is_fact: bool
    critic: Optional[list[str]] = None
    rubric: Optional[list[str]] = None

class ALFFunctionCall(BaseModel):
    name: str
    arguments: dict

class ALFLog(BaseModel):
    """
    A turn is a single interaction between a user and a bot.
    Determined by UID
    """
    uid: str
    time_in: str  # time of request
    time_out: Optional[str] = None  # time of response
    query: str
    summary: str
    with_knowledge: bool
    response_latency: Optional[float] = None
    response_type: str
    response: str
    references: Optional[list[str]] = None
    sent: bool = False
    fact_check: Optional[ALFFactCheck] = None
    function_call: Optional[ALFFunctionCall] = None
    language: Optional[str] = None
    
class ALFChat(BaseModel):
    """
    A chat is a single sequence of turns between a user and a bot.
    Determined by Chat ID
    """
    channel_id: str
    chat_id: str
    logs: list[ALFLog]
    feedback: Optional[str] = None