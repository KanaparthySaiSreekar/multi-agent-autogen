# =======================================================================
#                     chat/schemas/chat_schemas.py
# =======================================================================
# SIMPLIFIED: Only chat conversation and suggestion schemas

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# =======================================================================
#                         Chat Conversation Schemas
# =======================================================================


class ChatRequest(BaseModel):
    """The request model for sending a message to the agent."""

    message: str
    conversation_id: str | None = Field(
        None,
        description="ID for an ongoing conversation. If None, a new conversation starts.",
    )


class ChatHistoryResponse(BaseModel):
    """Response model for chat history."""

    history: List[dict]


class ConversationSummary(BaseModel):
    """Summary of a conversation for listing."""

    conversation_id: str
    last_updated_at: str
    chat_name: Optional[str] = "Chat"


# =======================================================================
#                         Autocomplete/Suggestions Schemas
# =======================================================================


class SuggestionRequest(BaseModel):
    """Incoming payload for autocomplete suggestions."""

    prefix: str = Field("", description="Current user input to complete.")
    max_suggestions: int = Field(
        5, ge=1, le=20, description="Max number of suggestions."
    )
    locale: Optional[str] = Field(
        None, description="Optional locale hint, e.g., 'en-US'."
    )


class SuggestionsResponse(BaseModel):
    """Structured response: list of suggestions."""

    suggestions: List[str] = Field(default_factory=list)


# =======================================================================
#                         Chat Naming Schemas
# =======================================================================


class ChatNameResponse(BaseModel):
    """Pydantic model for chat name generation response"""

    title: str = Field(
        ..., min_length=1, max_length=40, description="Generated chat title (2-4 words)"
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Clean and validate the title"""
        if not v:
            raise ValueError("Title cannot be empty")

        cleaned = v.strip().replace('"', "").replace("'", "")

        if len(cleaned) > 40:
            cleaned = cleaned[:40].rsplit(" ", 1)[0]

        common_only = {"chat", "conversation", "new", "untitled", "discussion"}
        if cleaned.lower() in common_only:
            raise ValueError("Title too generic")

        return cleaned.title()
