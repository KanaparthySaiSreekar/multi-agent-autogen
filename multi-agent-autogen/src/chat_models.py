from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ConversationSummary(BaseModel):
    id: str
    name: str
    updated_at: str
