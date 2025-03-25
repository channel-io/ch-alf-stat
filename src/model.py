
from pydantic import BaseModel
from typing import Optional


class AlfFactCheck(BaseModel):
    is_fact: bool
    critic: Optional[list[str]] = None
    rubric: Optional[list[str]] = None

class AlfFunctionCall(BaseModel):
    name: str
    arguments: dict

class AlfTurn(BaseModel):
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
    fact_check: Optional[AlfFactCheck] = None
    function_call: Optional[AlfFunctionCall] = None

class AlfLog(BaseModel):
    """
    A log is a single sequence of turns between a user and a bot.
    Determined by Chat ID
    """
    channel_id: str
    chat_id: str
    turns: list[AlfTurn]
    feedback: Optional[str] = None