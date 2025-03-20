from pydantic import BaseModel


class PongResponse(BaseModel):
    status: str = "pong"
