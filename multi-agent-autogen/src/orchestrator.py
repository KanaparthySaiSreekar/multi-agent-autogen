import asyncio
import json
import logging

from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent

from src.config import LLM_CONFIG
from src.models import ActionItem, AnalysisResult, RiskItem
from src.agents.summary_agent import create_summary_agent
from src.agents.action_agent import create_action_agent
from src.agents.risk_agent import create_risk_agent

logger = logging.getLogger(__name__)


async def _quick_llm_call(system_message: str, user_message: str) -> str:
    """Make a one-shot LLM call using initiate_chat (proven pattern)."""
    agent = AssistantAgent(
        name="Helper",
        system_message=system_message,
        llm_config=LLM_CONFIG,
        human_input_mode="NEVER",
    )
    user_proxy = UserProxyAgent(
        name="Caller",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )
    chat_result = await asyncio.to_thread(
        user_proxy.initiate_chat,
        agent,
        message=user_message,
        max_turns=1,
    )
    # Extract reply from chat history (last message from the agent)
    for msg in reversed(chat_result.chat_history):
        if msg.get("name") == "Helper":
            return msg.get("content", "")
    return ""


async def classify_intent(message: str) -> str:
    """Classify whether the message needs document analysis or is casual conversation."""
    system = (
        "You are a message classifier for a document analysis system. "
        "Determine if the user's message requires multi-agent document analysis "
        "or is casual conversation.\n\n"
        "Reply with ONLY one word:\n"
        "- 'analysis' if the message contains substantial text (meeting notes, reports, "
        "documents, articles, lengthy content) that should be analyzed for summaries, "
        "action items, and risks.\n"
        "- 'conversation' if the message is a greeting, question, casual chat, short request, "
        "or anything that doesn't contain a document to analyze.\n\n"
        "Reply with ONLY: analysis OR conversation"
    )
    reply = await _quick_llm_call(system, message)
    return "analysis" if "analysis" in reply.strip().lower() else "conversation"


async def simple_chat_streaming(message: str):
    """Handle casual conversation with a direct LLM response."""
    system = (
        "You are a friendly assistant for a Document Intelligence system. "
        "You help users with questions and casual conversation. "
        "If the user greets you, greet them back warmly. "
        "If they ask what you can do, explain that you can analyze documents, "
        "meeting notes, and reports to extract summaries, action items, and risks. "
        "They can paste or upload a document for analysis. "
        "Keep responses concise and helpful."
    )
    reply = await _quick_llm_call(system, message)
    yield {"type": "result", "component": None, "payload": None, "full_reply": reply}


async def route_message_streaming(message: str):
    """Classify user intent and route to the appropriate handler."""
    intent = await classify_intent(message)
    logger.info(f"Message classified as: {intent}")

    if intent == "analysis":
        async for event in analyze_document_streaming(message):
            yield event
    else:
        async for event in simple_chat_streaming(message):
            yield event


def _extract_json(text: str) -> dict | list | None:
    """Extract JSON from an agent response, handling markdown fences."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines[1:] if l.strip() != "```"]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object/array in the text
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        logger.warning(f"Could not parse JSON from agent response: {text[:200]}")
        return None


async def analyze_document(document_text: str) -> AnalysisResult:
    """
    Run the multi-agent pipeline to analyze a document.

    Pipeline: Summary Agent -> Action Agent -> Risk Agent
    Each agent sees all prior messages via GroupChat.
    """
    # 1. Create all three agents
    summary_agent = create_summary_agent(LLM_CONFIG)
    action_agent = create_action_agent(LLM_CONFIG)
    risk_agent = create_risk_agent(LLM_CONFIG)

    # 2. Create user proxy to kick off the conversation
    user_proxy = UserProxyAgent(
        name="Coordinator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # 3. GroupChat with sequential agent order
    group_chat = GroupChat(
        agents=[user_proxy, summary_agent, action_agent, risk_agent],
        messages=[],
        max_round=4,
        speaker_selection_method="round_robin",
    )
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=LLM_CONFIG,
    )

    # 4. Initiate the conversation (run in thread to not block FastAPI)
    prompt = (
        "Analyze the following document thoroughly. "
        "Each agent should analyze and produce structured JSON output.\n\n"
        f"--- DOCUMENT START ---\n{document_text}\n--- DOCUMENT END ---"
    )

    await asyncio.to_thread(
        user_proxy.initiate_chat,
        manager,
        message=prompt,
    )

    # 5. Parse each agent's response from chat history
    messages = group_chat.messages
    summary_data = {}
    action_data = {}
    risk_data = {}

    for msg in messages:
        name = msg.get("name", "")
        content = msg.get("content", "")
        if not content:
            continue

        parsed = _extract_json(content)
        if parsed is None:
            continue

        if name == "SummaryAgent":
            summary_data = parsed if isinstance(parsed, dict) else {}
        elif name == "ActionAgent":
            action_data = parsed if isinstance(parsed, dict) else {}
        elif name == "RiskAgent":
            risk_data = parsed if isinstance(parsed, dict) else {}

    # 6. Build the combined AnalysisResult
    action_items = []
    for item in action_data.get("action_items", []):
        try:
            action_items.append(ActionItem(**item))
        except Exception:
            logger.warning(f"Skipping malformed action item: {item}")

    risks = []
    for item in risk_data.get("risks", []):
        try:
            risks.append(RiskItem(**item))
        except Exception:
            logger.warning(f"Skipping malformed risk item: {item}")

    return AnalysisResult(
        summary=summary_data.get("summary", "No summary generated."),
        key_decisions=summary_data.get("key_decisions", []),
        action_items=action_items,
        risks=risks,
    )


async def analyze_document_streaming(document_text: str):
    """
    Async generator that runs the multi-agent pipeline and yields SSE events
    as each agent completes.
    """
    summary_agent = create_summary_agent(LLM_CONFIG)
    action_agent = create_action_agent(LLM_CONFIG)
    risk_agent = create_risk_agent(LLM_CONFIG)

    user_proxy = UserProxyAgent(
        name="Coordinator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    group_chat = GroupChat(
        agents=[user_proxy, summary_agent, action_agent, risk_agent],
        messages=[],
        max_round=4,
        speaker_selection_method="round_robin",
    )
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=LLM_CONFIG,
    )

    prompt = (
        "Analyze the following document thoroughly. "
        "Each agent should analyze and produce structured JSON output.\n\n"
        f"--- DOCUMENT START ---\n{document_text}\n--- DOCUMENT END ---"
    )

    agent_names = ["SummaryAgent", "ActionAgent", "RiskAgent"]
    seen_agents = set()

    # Yield initial status for the first agent
    yield {"type": "status", "agent": "SummaryAgent"}

    # Run the chat in a background thread
    chat_task = asyncio.ensure_future(
        asyncio.to_thread(user_proxy.initiate_chat, manager, message=prompt)
    )

    # Poll group_chat.messages for progress
    while not chat_task.done():
        await asyncio.sleep(0.5)
        for msg in group_chat.messages:
            name = msg.get("name", "")
            if name in agent_names and name not in seen_agents:
                seen_agents.add(name)
                yield {"type": "agent_done", "agent": name}
                # Signal next agent starting
                idx = agent_names.index(name)
                if idx + 1 < len(agent_names):
                    yield {"type": "status", "agent": agent_names[idx + 1]}

    # Ensure task exceptions propagate
    try:
        await chat_task
    except Exception as e:
        logger.exception("Agent chat failed")
        yield {"type": "error", "content": str(e)}
        return

    # Final check for any agents we missed during polling
    for msg in group_chat.messages:
        name = msg.get("name", "")
        if name in agent_names and name not in seen_agents:
            seen_agents.add(name)
            yield {"type": "agent_done", "agent": name}

    # Parse results (reuse same logic as analyze_document)
    messages = group_chat.messages
    summary_data = {}
    action_data = {}
    risk_data = {}

    for msg in messages:
        name = msg.get("name", "")
        content = msg.get("content", "")
        if not content:
            continue
        parsed = _extract_json(content)
        if parsed is None:
            continue
        if name == "SummaryAgent":
            summary_data = parsed if isinstance(parsed, dict) else {}
        elif name == "ActionAgent":
            action_data = parsed if isinstance(parsed, dict) else {}
        elif name == "RiskAgent":
            risk_data = parsed if isinstance(parsed, dict) else {}

    action_items = []
    for item in action_data.get("action_items", []):
        try:
            action_items.append(ActionItem(**item))
        except Exception:
            logger.warning(f"Skipping malformed action item: {item}")

    risks = []
    for item in risk_data.get("risks", []):
        try:
            risks.append(RiskItem(**item))
        except Exception:
            logger.warning(f"Skipping malformed risk item: {item}")

    result = AnalysisResult(
        summary=summary_data.get("summary", "No summary generated."),
        key_decisions=summary_data.get("key_decisions", []),
        action_items=action_items,
        risks=risks,
    )

    payload = result.model_dump()
    full_reply = payload.get("summary", "")

    yield {"type": "result", "component": "analysis_card", "payload": payload, "full_reply": full_reply}
