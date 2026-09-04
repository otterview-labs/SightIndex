import uuid
from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    session_id: uuid.UUID | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class ChatToolCall(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    data: dict[str, Any] = Field(default_factory=dict)
    tool_name: str | None = None
    tool_params: dict[str, Any] | None = None
    tool_calls: list[ChatToolCall] = Field(default_factory=list)
