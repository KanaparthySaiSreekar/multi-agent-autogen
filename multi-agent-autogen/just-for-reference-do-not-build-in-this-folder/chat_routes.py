# =======================================================================
#                     chat/routes/chat_routes.py
# =======================================================================
# Chat routes using thread-safe AgentManager

from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.agents_manager import BaseAgentManager as AgentManager, get_agent_manager
from app.auth.api.deps import get_current_active_user
from app.chat.schemas.chat_schemas import (
    ChatRequest,
    ConversationSummary,
    SuggestionRequest,
    SuggestionsResponse,
)
from app.chat.services.chat_service import ChatService
from app.chat.services.suggestions_service import generate_suggestions
from app.db.engine import AsyncSessionLocal
from app.db.models import User

router = APIRouter(prefix="/api/chat", tags=["Chat"])


async def get_db():
    """Database dependency."""
    async with AsyncSessionLocal() as session:
        yield session


async def get_agent_manager_dep() -> AgentManager:
    """Dependency to get the AgentManager singleton."""
    return await get_agent_manager()


# =======================================================================
#                    Chat Conversation Endpoints (ENHANCED)
# =======================================================================


@router.post("/")
async def chat_stream_endpoint(
    request: ChatRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: AsyncSession = Depends(get_db),
    agent_manager: AgentManager = Depends(get_agent_manager_dep),
):
    """Stream chat responses with user context."""
    if not agent_manager.is_initialized:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # Validate recruiter context for recruiters
    if current_user.role.value == "RECRUITER":
        if not current_user.recruiter:
            raise HTTPException(
                status_code=400,
                detail="Recruiter user must have associated recruiter profile",
            )

    return StreamingResponse(
        ChatService.stream_chat_generator(
            request, db, agent_manager, current_user=current_user
        ),
        media_type="text/event-stream",
    )


@router.patch("/set_name/{conversation_id}")
async def set_chat_name(
    conversation_id: str,
    chat_name: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: AsyncSession = Depends(get_db),
):
    """Set or update the name of a chat conversation."""
    if not conversation_id:
        raise HTTPException(status_code=400, detail="Conversation ID is required.")
    if not chat_name or len(chat_name) > 255:
        raise HTTPException(
            status_code=400, detail="Chat name must be between 1 and 255 characters."
        )

    try:
        await ChatService.set_chat_name(
            conversation_id, chat_name, db, user_id=current_user.user_id
        )
        return {"message": "Chat name updated successfully."}
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}/history")
async def get_chat_history(
    conversation_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: AsyncSession = Depends(get_db),
):
    """Get chat history for a conversation."""
    if not conversation_id:
        raise HTTPException(status_code=400, detail="Conversation ID is required.")

    try:
        history = await ChatService.get_chat_history(
            conversation_id, db, user_id=current_user.user_id
        )
        return history
    except ValueError as e:
        if "Invalid conversation_id format" in str(e):
            raise HTTPException(
                status_code=400, detail="Invalid conversation_id format."
            )
        elif "Conversation not found" in str(e):
            raise HTTPException(status_code=404, detail="Conversation not found.")
        else:
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def get_all_chats(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: AsyncSession = Depends(get_db),
) -> List[ConversationSummary]:
    """
    Retrieves all distinct chat conversations for the current user,
    ordered by the most recently updated.
    """
    conversations = await ChatService.get_all_conversations(
        db, user_id=current_user.user_id
    )
    return [ConversationSummary(**conv) for conv in conversations]


# =======================================================================
#                    Autocomplete Suggestions Endpoint (UNCHANGED)
# =======================================================================


@router.post(
    "/suggestions",
    response_model=SuggestionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get autocomplete suggestions",
)
async def get_suggestions(payload: SuggestionRequest) -> SuggestionsResponse:
    """
    Returns short, relevant autocomplete suggestions for a given prefix.
    """
    return await generate_suggestions(payload)
