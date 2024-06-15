from pydantic import BaseModel


class TraceResponse(BaseModel):
    status: str
