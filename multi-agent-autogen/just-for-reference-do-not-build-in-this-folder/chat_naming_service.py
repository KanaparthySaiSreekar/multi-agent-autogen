from typing import List, Dict
from loguru import logger

from app.chat.schemas.chat_schemas import ChatNameResponse
from app.clients.schemas import ChatOptions


class SimpleChatNamingService:
    """Simple, effective chat naming using 3 messages + LLM with validation"""

    @staticmethod
    async def generate_chat_name(messages: List[Dict]) -> str:
        """
        Generate chat name using first 3 meaningful messages with Pydantic validation.

        Args:
            messages: Chat history with user/assistant messages

        Returns:
            str: Generated chat name (2-4 words)
        """

        params = ChatOptions(
            extra_params={
                "temperature": 0.3,
                "top_p": 0.9,
                "max_output_tokens": 25,
            }
        )

        if len(messages) < 3:
            return "Chat"

        user_messages = [
            msg["content"] for msg in messages if msg.get("role") == "user"
        ]

        if len(user_messages) < 3:
            return "Chat"

        greeting = user_messages[0]
        substantial = user_messages[1]

        try:
            from app.clients.clients import az_client
            from app.config import settings

            system_prompt = """Generate a short 2-4 word title for this chat conversation.
Focus on the main request or topic. Be concise and descriptive.

Examples:
- "Java Developer Job"
- "React Candidate Search"  
- "Salary Market Data"
- "Python Interview Questions"

Respond with a JSON object containing only the title field.
Do not include quotes around the title value itself."""

            user_prompt = f"""First message: {greeting}
Second message: {substantial}

Generate a short title for this conversation."""

            response = await az_client.call_llm(
                model_name=settings.ai.AZURE_OPENAI_DEPLOYMENT_41,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                validation_class=ChatNameResponse,
                params=params,
            )

            return response.title

        except Exception as e:
            logger.warning(f"LLM chat naming failed: {e}")

            return SimpleChatNamingService._fallback_naming(substantial)

    @staticmethod
    def _fallback_naming(message: str) -> str:
        """Simple fallback if LLM fails"""
        import re

        common_words = {
            "job",
            "role",
            "position",
            "find",
            "create",
            "need",
            "want",
            "looking",
            "for",
        }
        words = re.findall(r"\b[A-Za-z]{3,}\b", message)
        meaningful = [w.title() for w in words[:5] if w.lower() not in common_words]

        if meaningful:
            return " ".join(meaningful[:3])
        return "Chat"

    @staticmethod
    def should_generate_name(messages: List[Dict], current_name: str) -> bool:
        """
        Check if we should generate a name for this chat.

        Args:
            messages: Current chat messages
            current_name: Current chat name

        Returns:
            bool: True if should generate name
        """

        if current_name != "Chat":
            return False

        user_messages = [
            msg["content"] for msg in messages if msg.get("role") == "user"
        ]

        return len(user_messages) >= 3