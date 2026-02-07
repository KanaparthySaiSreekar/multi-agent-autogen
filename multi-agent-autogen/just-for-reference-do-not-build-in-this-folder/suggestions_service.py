# =======================================================================
#                     chat/services/suggestions_service.py
# =======================================================================
# KEEP EXISTING - Autocomplete suggestions service (no changes needed)

import asyncio
from loguru import logger

from app.chat.schemas.chat_schemas import SuggestionRequest, SuggestionsResponse
from app.clients.schemas import ChatOptions
from app.clients.clients import az_client
from app.agents.prompts import AUTO_COMPLETE_PROMPT
from app.config import settings


def _build_user_prompt(req: SuggestionRequest) -> str:
    """Build user prompt for autocomplete suggestions"""
    base = f'prefix="{req.prefix.strip()}"\nmax={req.max_suggestions}'
    if req.locale:
        base += f'\nlocale="{req.locale}"'
    return base


async def generate_suggestions(req: SuggestionRequest) -> SuggestionsResponse:
    """
    Generate suggestions using the LLM with strict structured output.
    Falls back to defaults when prefix is empty.
    """
    params = ChatOptions(
        extra_params={
            "temperature": 0.3,
            "top_p": 0.9,
            "max_output_tokens": 150,
        }
    )

    try:
        response: SuggestionsResponse = await az_client.call_llm(
            model_name=settings.ai.AZURE_OPENAI_DEPLOYMENT_41,
            user_prompt=_build_user_prompt(req),
            system_prompt=AUTO_COMPLETE_PROMPT,
            validation_class=SuggestionsResponse,
            params=params,
        )

        # Clean and deduplicate suggestions
        cleaned = []
        seen = set()
        for suggestion in response.suggestions:
            suggestion_clean = suggestion.strip()
            if suggestion_clean and suggestion_clean not in seen:
                cleaned.append(suggestion_clean)
                seen.add(suggestion_clean)
            if len(cleaned) >= req.max_suggestions:
                break

        return SuggestionsResponse(suggestions=cleaned)

    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.info("LLM suggestion generation failed, falling back to defaults")
        # Could add fallback default suggestions here if needed
        raise Exception("LLM suggestion generation failed", str(e))
