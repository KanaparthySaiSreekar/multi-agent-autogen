# =======================================================================
#                     chat/services/chat_service.py
# =======================================================================
# Chat service using thread-safe AgentManager

import json
import uuid
import tiktoken
from typing import AsyncGenerator, Optional, Dict, Any
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from autogen_agentchat.messages import (
    ToolCallExecutionEvent,
    ModelClientStreamingChunkEvent,
    TextMessage,
)
from autogen_core.model_context import BufferedChatCompletionContext

from app.chat.services.chat_naming_service import SimpleChatNamingService
from app.db.models import UserChatHistory, User
from app.db.schemas import UserRole
from app.agents.agents_manager import (
    BaseAgentManager as AgentManager,
    CONVERSATION_BUFFER_SIZE,
)
from app.chat.schemas.chat_schemas import ChatRequest


class ChatService:
    """Service for handling chat operations and conversation management with user context."""

    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4") -> int:
        """
        Count the number of tokens in a text string using tiktoken.

        Args:
            text: The text to count tokens for
            model: The model name (default: gpt-4)

        Returns:
            Number of tokens in the text
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base encoding for GPT-4/GPT-3.5-turbo
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

    @staticmethod
    async def stream_chat_generator(
        request: ChatRequest,
        db: AsyncSession,
        agent_manager: AgentManager,
        current_user: Optional[User] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Handles the agent conversation with proper user context.
        Uses thread-safe AgentManager for agent creation.
        """
        conv_id_str = request.conversation_id

        if not agent_manager.is_initialized:
            raise RuntimeError("Server is not ready, AgentManager not initialized.")

        # Create fresh model context for this conversation
        fresh_model_context = BufferedChatCompletionContext(
            buffer_size=CONVERSATION_BUFFER_SIZE
        )

        # Get enhanced config with user context
        agent_config = agent_manager.get_agent_config()
        enhanced_agent_config = ChatService._enhance_agent_config_with_user_context(
            agent_config, current_user
        )

        saved_chat_obj = None
        chat_history = []
        conv_id_uuid = None

        # Use current user ID if available, otherwise default
        user_id = current_user.user_id

        # Load chat FIRST to check for job associations BEFORE creating agent
        if conv_id_str:
            logger.info(f"Attempting to resume conversation: {conv_id_str}")
            try:
                conv_id_uuid = uuid.UUID(conv_id_str)
                stmt = select(UserChatHistory).where(
                    UserChatHistory.chat_id == conv_id_uuid,
                    UserChatHistory.user_id == user_id,  # NEW: Filter by user
                )
                result = await db.execute(stmt)
                saved_chat_obj = result.scalar_one_or_none()
            except ValueError:
                error_data = {
                    "type": "error",
                    "content": f"Invalid conversation_id format: {conv_id_str}",
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                return

            if saved_chat_obj:
                logger.success(
                    f"Successfully loaded state for conversation: {conv_id_str} (user: {user_id})"
                )

                if saved_chat_obj.chat_details and isinstance(
                    saved_chat_obj.chat_details, list
                ):
                    chat_history = list(saved_chat_obj.chat_details)

                # Add job context to system message BEFORE creating agent
                if saved_chat_obj.job_draft_id or saved_chat_obj.job_posting_id:
                    job_context = "\n\n# Job Association Warning\n"
                    if saved_chat_obj.job_posting_id:
                        job_context += f"This chat already has a posted job (Job ID: {saved_chat_obj.job_posting_id}).\n"
                    elif saved_chat_obj.job_draft_id:
                        job_context += f"This chat already has a job draft (Draft ID: {saved_chat_obj.job_draft_id}).\n"
                    job_context += """
CRITICAL: Each chat can only manage ONE job. If the user asks to create another job, politely inform them:
"This chat is already managing a job. To create a new job, please start a new chat conversation."
Do NOT proceed with creating another job in this chat.
"""
                    enhanced_agent_config["system_message"] = (
                        enhanced_agent_config.get("system_message", "") + job_context
                    )
                    logger.info(
                        f"Added job enforcement context to agent system message for chat {conv_id_str}"
                    )
            else:
                logger.warning(
                    f"Conversation ID {conv_id_str} not found for user {user_id}. Starting new conversation."
                )

        # NOW create agent with the complete system message (including job context if applicable)
        agent = await agent_manager.create_agent_instance(
            model_context=fresh_model_context,
            custom_system_message=enhanced_agent_config.get("system_message"),
        )

        # Load agent state if resuming conversation
        if saved_chat_obj and saved_chat_obj.agent_state:
            await agent.load_state(saved_chat_obj.agent_state)

        if not saved_chat_obj:
            if not conv_id_uuid:
                conv_id_uuid = uuid.uuid4()
                conv_id_str = str(conv_id_uuid)
            logger.info(f"Starting new conversation: {conv_id_str} for user: {user_id}")
            saved_chat_obj = UserChatHistory(
                chat_id=conv_id_uuid,
                user_id=user_id,  # NEW: Use actual user ID
                agent_state={},
                chat_details=[],
            )

        # Log only user message (concise)
        logger.info(
            f"User message (conv: {conv_id_str[:8]}...): {request.message[:100]}{'...' if len(request.message) > 100 else ''}"
        )
        chat_history.append({"role": "user", "content": request.message})

        try:
            stream = agent.run_stream(task=request.message)
            prompt_tokens_used = 0
            completion_tokens_used = 0

            final_message_history = []
            async for message in stream:
                final_message_history.append(message)
                if isinstance(message, ModelClientStreamingChunkEvent):
                    chunk = message.content
                    data_to_send = {"type": "chunk", "content": chunk}
                    yield f"data: {json.dumps(data_to_send)}\n\n"

                    if getattr(message, "models_usage", None) is not None:
                        usage = message.models_usage
                        prompt_tokens_used = usage.prompt_tokens or 0
                        completion_tokens_used = usage.completion_tokens or 0

                usage = getattr(message, "models_usage", None)
                if usage is not None:
                    prompt_tokens_used = usage.prompt_tokens or 0
                    completion_tokens_used = usage.completion_tokens or 0

            full_reply_text = ""
            payload = None
            component_name = None

            if final_message_history:
                for message in reversed(final_message_history):
                    if isinstance(message, TextMessage):
                        full_reply_text = message.content
                        break

                # Tool to component mapping
                TOOL_TO_COMPONENT_MAP = {
                    "show_recruiter_welcome_card": "recruiter_welcome_card",
                    "get_assigned_job_requirements": "existing_cases",
                    "list_recruiter_clients": "client_selector",
                    "create_job_from_text": "job_description",
                    "create_job_draft_from_requirement": "job_description",
                    "create_job_requirement_with_draft": "job_description",
                    "get_sourced_candidates": "sourced_candidates",
                    "create_default_pipeline_for_draft": "draft_pipeline",
                    "get_draft_pipeline": "drafted_pipeline",
                    "configure_draft_ai_interview": "ai_interview_config",
                    "configure_draft_technical_assessment": "technical_assessment_config",
                    "create_job_posting": "job_posted",
                    "get_job_pipeline": "job_pipeline",
                    "get_draft_questions": "prescreening_questions",
                    "create_job_posting_from_draft": "job_posted_draft",
                    "post_job_to_ceipal": "ceipal_posting",
                    "get_ranked_candidates": "ranked_candidates",
                    "create_job_pipeline": "pipeline_configuration",
                    "get_job_pipeline_detailed": "pipeline_details",
                    "move_candidates_to_stage_simple": "candidate_movement",
                    "get_job_candidates_by_stage": "applications_stage",
                    "get_default_pipeline_template": "pipeline_template",
                    "get_pipeline_statistics": "pipeline_analytics",
                    "edit_job_draft": "edit_job_draft",
                }

                for message in reversed(final_message_history):
                    if isinstance(message, ToolCallExecutionEvent):
                        tool_results = message.content
                        if not tool_results:
                            continue

                        last_tool_result = tool_results[-1]
                        tool_name = last_tool_result.name

                        if tool_name in TOOL_TO_COMPONENT_MAP:
                            component_name = TOOL_TO_COMPONENT_MAP[tool_name]
                            try:
                                # CRITICAL FIX: Validate content is not empty before parsing
                                if not last_tool_result.content:
                                    raise ValueError("Tool returned empty content")

                                if isinstance(last_tool_result.content, str) and last_tool_result.content.strip() == "":
                                    raise ValueError("Tool returned empty string")

                                data = json.loads(last_tool_result.content)

                                # Validate data structure
                                if not data:
                                    raise ValueError("Tool returned empty JSON")

                                # Check for nested JSON string format
                                if isinstance(data, list) and len(data) > 0:
                                    first_item = data[0]
                                    if isinstance(first_item, dict) and "text" in first_item:
                                        text_content = first_item["text"]
                                        if not text_content or (isinstance(text_content, str) and text_content.strip() == ""):
                                            raise ValueError("Tool returned empty text field")
                                        payload = json.loads(text_content)
                                    else:
                                        # First item is a dict but no "text" field, use it directly
                                        payload = first_item
                                elif isinstance(data, dict):
                                    # Data is already a dict, use it directly
                                    payload = data
                                else:
                                    # Data is some other type, wrap it
                                    payload = {"result": data}

                            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                                error_msg = f"Tool '{tool_name}' returned invalid response: {type(e).__name__}: {str(e)}"
                                logger.error(f"{error_msg}. Raw content: {last_tool_result.content[:500] if last_tool_result.content else 'None'}")
                                payload = {
                                    "error": f"Failed to parse tool payload for '{tool_name}': {str(e)}"
                                }
                            break

            assistant_turn = {
                "role": "assistant",
                "content": full_reply_text,
                "component": component_name,
                "payload": payload,
            }
            chat_history.append(assistant_turn)

            # NEW: Link job draft or job posting to chat based on tool execution
            if payload and isinstance(payload, dict):
                # Link draft_id when job draft is created
                # Note: GeneratedJD schema uses 'process_id' which is the draft_id
                if component_name == "job_description":
                    draft_id_str = payload.get("draft_id") or payload.get("process_id")
                    if draft_id_str:
                        try:
                            # Convert to string if it's a UUID object
                            if not isinstance(draft_id_str, str):
                                draft_id_str = str(draft_id_str)
                            saved_chat_obj.job_draft_id = uuid.UUID(draft_id_str)
                            logger.info(
                                f"Linked draft {draft_id_str} to chat {conv_id_str}"
                            )
                        except (ValueError, AttributeError, TypeError) as e:
                            logger.warning(f"Failed to link draft_id to chat: {e}")

                # Link job_posting_id when job is posted
                elif (
                    component_name in ["job_posted", "job_posted_draft"]
                    and "job_id" in payload
                ):
                    job_id_str = payload.get("job_id")
                    if job_id_str:
                        try:
                            # Convert to string if it's a UUID object
                            if not isinstance(job_id_str, str):
                                job_id_str = str(job_id_str)
                            saved_chat_obj.job_posting_id = uuid.UUID(job_id_str)
                            saved_chat_obj.job_draft_id = None
                            logger.info(
                                f"Linked job posting {job_id_str} to chat {conv_id_str} and cleared up the draft reference."
                            )
                        except (ValueError, AttributeError, TypeError) as e:
                            logger.warning(
                                f"Failed to link job_posting_id to chat: {e}"
                            )

            new_agent_state_json = await agent.save_state()

            saved_chat_obj.agent_state = new_agent_state_json
            saved_chat_obj.chat_details = chat_history

            try:
                if SimpleChatNamingService.should_generate_name(
                    chat_history, saved_chat_obj.chat_name or "Chat"
                ):
                    logger.info(f"Auto-naming chat {conv_id_str} after 2nd message...")

                    new_name = await SimpleChatNamingService.generate_chat_name(
                        chat_history
                    )
                    saved_chat_obj.chat_name = new_name

                    logger.success(f"Chat {conv_id_str} named: '{new_name}'")

            except Exception as e:
                logger.warning(f"Auto-naming failed for {conv_id_str}: {e}")

            db.add(saved_chat_obj)
            await db.commit()
            logger.success(
                f"Successfully saved state and history for conversation: {conv_id_str} (user: {user_id})"
            )
            logger.info(
                f"Prompt Tokens Used: {prompt_tokens_used}\nCompletion Tokens used: {completion_tokens_used}"
            )
            if prompt_tokens_used or completion_tokens_used:
                input_tokens = prompt_tokens_used
                output_tokens = completion_tokens_used
                logger.info(
                    "[LLM USAGE] conv={conv} user={user} source=provider "
                    "input_tokens={inp} output_tokens={out} total_tokens={tot}",
                    conv=conv_id_str,
                    user=str(user_id),
                    inp=input_tokens,
                    out=output_tokens,
                    tot=input_tokens + output_tokens,
                )
            else:
                logger.info(
                    "[LLM USAGE] conv={conv} user={user} source=tiktoken (provider usage missing)",
                    conv=conv_id_str,
                    user=str(user_id),
                )
                input_tokens = ChatService.count_tokens(request.message)
                output_tokens = ChatService.count_tokens(full_reply_text)

            total_tokens = input_tokens + output_tokens

            logger.info(
                "[LLM USAGE SUMMARY] conv={conv} user={user} input_tokens={inp} "
                "output_tokens={out} total_tokens={tot}",
                conv=conv_id_str,
                user=str(user_id),
                inp=input_tokens,
                out=output_tokens,
                tot=total_tokens,
            )

            end_data = {
                "type": "end",
                "conversation_id": conv_id_str,
                "component": component_name,
                "payload": payload,
                "full_reply": full_reply_text,
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            }
            yield f"data: {json.dumps(end_data)}\n\n"

        except Exception as e:
            logger.exception(
                f"An unhandled error occurred in stream_chat_generator for conversation {conv_id_str}"
            )
            error_data = {
                "type": "error",
                "content": str(e),
                "conversation_id": conv_id_str,
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    @staticmethod
    def _enhance_agent_config_with_user_context(
        agent_config: Dict[str, Any], current_user: Optional[User]
    ) -> Dict[str, Any]:
        """
        NEW: Enhance agent config with user context for job creation.
        Injects user information into the system message.
        """
        if not current_user:
            return agent_config

        enhanced_config = agent_config.copy()
        original_system_message = enhanced_config.get("system_message", "")

        # Add user context to system message
        user_context = f"""

# Current User Context
You are acting on behalf of:
- User ID: {current_user.user_id}
- Full Name: {current_user.full_name}
- Email: {current_user.email}
- Role: {current_user.role.value}
"""

        # Add recruiter context if user is a recruiter
        if current_user.role == UserRole.RECRUITER and current_user.recruiter:
            user_context += f"- Recruiter ID: {current_user.recruiter.recruiter_id}\n"
            user_context += """
IMPORTANT: When creating jobs, use this recruiter's ID instead of creating default recruiters.
The job creation tools should use the current user's recruiter_id for proper attribution.
"""

        user_context += """
This context is for system use only - do not mention these technical details to the user.
Proceed with the user's request naturally.
"""

        enhanced_config["system_message"] = original_system_message + user_context

        logger.debug(
            f"Enhanced agent config with context for user: {current_user.full_name} ({current_user.role.value})"
        )

        return enhanced_config

    @staticmethod
    async def get_chat_history(
        conversation_id: str, db: AsyncSession, user_id: Optional[uuid.UUID] = None
    ) -> list:
        """Get chat history for a conversation with optional user filtering."""
        try:
            conv_id_uuid = uuid.UUID(conversation_id)
        except ValueError:
            raise ValueError("Invalid conversation_id format.")

        stmt = select(UserChatHistory).where(UserChatHistory.chat_id == conv_id_uuid)

        # NEW: Filter by user if provided
        if user_id:
            stmt = stmt.where(UserChatHistory.user_id == user_id)

        result = await db.execute(stmt)
        saved_chat = result.scalar_one_or_none()

        if not saved_chat or not saved_chat.chat_details:
            raise ValueError("Conversation not found or not accessible.")

        return saved_chat.chat_details

    @staticmethod
    async def get_all_conversations(
        db: AsyncSession, user_id: Optional[uuid.UUID] = None
    ) -> list:
        """Get all conversations for a user, ordered by most recent."""
        from sqlalchemy import func

        # NEW: Filter by user if provided
        base_query = select(UserChatHistory)
        if user_id:
            base_query = base_query.where(UserChatHistory.user_id == user_id)

        subq = (
            base_query.with_only_columns(
                UserChatHistory.chat_id,
                func.max(UserChatHistory.updated_at).label("last_updated_at"),
                UserChatHistory.chat_name,
            )
            .group_by(UserChatHistory.chat_id, UserChatHistory.chat_name)
            .subquery("latest_updates")
        )

        stmt = select(
            subq.c.chat_id, subq.c.last_updated_at, subq.c.chat_name
        ).order_by(subq.c.last_updated_at.desc())

        result = await db.execute(stmt)
        conversations = result.all()

        if not conversations:
            return []

        formatted_conversations = [
            {
                "conversation_id": str(conv.chat_id),
                "last_updated_at": conv.last_updated_at.isoformat(),
                "chat_name": conv.chat_name,
            }
            for conv in conversations
        ]

        logger.info(
            f"Retrieved {len(formatted_conversations)} conversations for user: {user_id}"
        )
        return formatted_conversations

    @staticmethod
    async def set_chat_name(
        conversation_id: str,
        chat_name: str,
        db: AsyncSession,
        user_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Set or update the name of a chat conversation with user filtering."""
        try:
            conversation_id = uuid.UUID(conversation_id)
        except ValueError:
            raise ValueError("Invalid conversation_id format.")

        stmt = select(UserChatHistory).where(UserChatHistory.chat_id == conversation_id)

        # NEW: Filter by user if provided
        if user_id:
            stmt = stmt.where(UserChatHistory.user_id == user_id)

        result = await db.execute(stmt)
        saved_chat = result.scalar_one_or_none()

        if not saved_chat:
            raise ValueError("Conversation not found or not accessible.")

        if saved_chat.chat_name == chat_name:
            raise ValueError("Chat name is already set to the provided value.")

        saved_chat.chat_name = chat_name
        db.add(saved_chat)
        await db.commit()

        logger.info(
            f"Updated chat name for conversation {conversation_id} to: {chat_name}"
        )
