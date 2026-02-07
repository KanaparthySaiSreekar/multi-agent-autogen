# =======================================================================
#                     agents/agents_manager.py
# =======================================================================
# Unified agent manager with shared base class for all agent types.
# Optimized with shared MCP sessions for connection pooling and fast shutdown.

import asyncio
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from loguru import logger
from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_ext.tools.mcp import (
    StreamableHttpServerParams,
    create_mcp_server_session,
    mcp_server_tools,
)

from app.config import settings
from app.clients.clients import agent_az_client
from app.agents.prompts import (
    CHATPROMPT,
    SALES_SYSTEM_PROMPT,
    CLIENT_SYSTEM_PROMPT,
    CANDIDATE_SYSTEM_PROMPT,
)


# =======================================================================
# Configuration
# =======================================================================

MCP_SERVER_URL = settings.MCP_SERVER_URL
CONVERSATION_BUFFER_SIZE = 15


class AgentType(Enum):
    """Enumeration of available agent types."""

    RECRUITER = "recruiter"
    SALES = "sales"
    CLIENT = "client"
    CANDIDATE = "candidate"


class AgentConfig(BaseModel):
    """Configuration for an agent type."""

    model_config = {"arbitrary_types_allowed": True}

    agent_type: AgentType
    name: str
    system_prompt: str
    tool_names: List[str]
    max_concurrent_agents: int = 50
    context_fields: List[str] = Field(default_factory=list)
    role_name: str = ""


# =======================================================================
# Agent Configurations Registry
# =======================================================================

AGENT_CONFIGS: Dict[AgentType, AgentConfig] = {
    AgentType.RECRUITER: AgentConfig(
        agent_type=AgentType.RECRUITER,
        name="HiringOrchestratorAgent",
        system_prompt=CHATPROMPT,
        max_concurrent_agents=50,
        context_fields=[],
        role_name="Recruiter",
        tool_names=[
            "show_recruiter_welcome_card",
            "list_recruiter_clients",
            "create_job_requirement_with_draft",
            "get_assigned_job_requirements",
            "create_job_draft_from_requirement",
            "create_job_from_text",
            "get_draft_questions",
            "create_ceipal_client",
            "verify_client_in_ceipal",
            "create_default_pipeline_for_draft",
            "get_draft_pipeline",
            "update_draft_pipeline_stage_questions",
            "add_draft_pipeline_stage",
            "remove_draft_pipeline_stage",
            "reorder_draft_pipeline_stages",
            "update_draft_stage_configuration",
            "toggle_draft_stage_active",
            "configure_draft_ai_interview",
            "get_draft_technical_assessments",
            "configure_draft_technical_assessment",
            "edit_job_draft",
            "create_job_posting_from_draft",
            "get_job_candidates_by_stage",
            "send_prescreening_invites",
            "get_job_pipeline",
            "move_candidates_to_stage_simple",
        ],
    ),
    AgentType.SALES: AgentConfig(
        agent_type=AgentType.SALES,
        name="SalesOrchestratorAgent",
        system_prompt=SALES_SYSTEM_PROMPT,
        max_concurrent_agents=30,
        context_fields=["sales_id", "user_id"],
        role_name="Sales",
        tool_names=[
            "show_sales_welcome_card",
            "trigger_job_requirement_form",
            "sales_create_job_from_text",
            "sales_create_job_requirement_from_jd_draft",
            "get_sales_job_requirements",
            "get_available_recruiters_for_sales",
        ],
    ),
    AgentType.CLIENT: AgentConfig(
        agent_type=AgentType.CLIENT,
        name="HiringOrchestratorAgent",
        system_prompt=CLIENT_SYSTEM_PROMPT,
        max_concurrent_agents=40,
        context_fields=["client_id", "user_id"],
        role_name="Client",
        tool_names=[
            "show_client_welcome_card",
            "trigger_client_job_requirement_form",
            "create_job_from_text",
            "create_job_requirement_from_jd_draft",
        ],
    ),
    AgentType.CANDIDATE: AgentConfig(
        agent_type=AgentType.CANDIDATE,
        name="HiringOrchestratorAgent",
        system_prompt=CANDIDATE_SYSTEM_PROMPT,
        max_concurrent_agents=30,
        context_fields=["candidate_id", "user_id"],
        role_name="Candidate",
        tool_names=[
            "show_candidate_welcome_card",
            "answer_candidate_question",
            # New dedicated tools for reliable, pre-built queries
            "get_candidate_interview_results",
            "get_candidate_assessment_results",
            "get_candidate_application_timeline",
            "get_candidate_recruiter_contact",
            "get_candidate_offer_details",
            "get_candidate_pipeline_progress",
        ],
    ),
}


# =======================================================================
# Context Injection Template
# =======================================================================


def build_context_injection(role: str, **context_fields) -> str:
    """Build a standardized context injection string for system messages."""
    if not context_fields:
        return ""

    context_lines = "\n".join(
        f"- {key.replace('_', ' ').title()}: {value}"
        for key, value in context_fields.items()
        if value is not None
    )

    return f"""

==========================================
USER CONTEXT
==========================================

{context_lines}
- Role: {role}

This context is automatically available to all tools. When calling tools that require IDs,
use the IDs shown above EXACTLY as provided (do not modify or validate them).

==========================================
"""


# =======================================================================
# Shared MCP Session Manager
# =======================================================================


class MCPSessionManager:
    """
    Manages a shared MCP session with connection pooling.
    All agents share a single session to the MCP server for efficient
    connection management and fast shutdown.
    """

    _instance: Optional["MCPSessionManager"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._session = None
        self._session_context = None
        self._all_tools: Dict[str, Any] = {}  # tool_name -> tool adapter
        self._initialized = False
        self._server_params: Optional[StreamableHttpServerParams] = None

    @classmethod
    async def get_instance(cls) -> "MCPSessionManager":
        """Get or create the singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = MCPSessionManager()
        return cls._instance

    async def initialize(self) -> None:
        """
        Initialize the shared MCP session and load all tools at once.
        This creates a single connection that's reused by all agents.
        """
        if self._initialized:
            logger.warning("MCPSessionManager already initialized")
            return

        logger.info("Initializing shared MCP session...")

        self._server_params = StreamableHttpServerParams(
            url=f"{MCP_SERVER_URL}/mcp",
            timeout=90.0,  # INCREASED FROM 30s: Allow more time under load to prevent timeout errors
            sse_read_timeout=300.0,
            terminate_on_close=True,
        )

        try:
            # Create and enter the session context
            self._session_context = create_mcp_server_session(self._server_params)
            self._session = await self._session_context.__aenter__()
            await self._session.initialize()

            # Get ALL tools from the server in one call
            all_tools = await mcp_server_tools(
                server_params=self._server_params, session=self._session
            )

            # Index tools by name for quick lookup
            for tool in all_tools:
                self._all_tools[tool.name] = tool

            self._initialized = True
            logger.success(
                f"MCPSessionManager initialized with {len(self._all_tools)} tools available"
            )
            logger.info(f"Available tools: {list(self._all_tools.keys())}")

        except Exception as e:
            logger.error(f"Failed to initialize MCP session: {e}")
            logger.exception("Full traceback:")
            raise

    def get_tools(self, tool_names: List[str]) -> List[Any]:
        """
        Get specific tools by name from the shared session.
        Returns only the tools that exist.
        """
        if not self._initialized:
            raise RuntimeError("MCPSessionManager not initialized")

        tools = []
        missing = []
        for name in tool_names:
            if name in self._all_tools:
                tools.append(self._all_tools[name])
            else:
                missing.append(name)

        if missing:
            logger.warning(f"Tools not found in MCP server: {missing}")

        return tools

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the shared MCP session with timeout protection.
        This closes the single connection cleanly.
        """
        if not self._initialized:
            return

        logger.info("Shutting down shared MCP session...")

        try:
            if self._session_context:
                # Exit the context manager with timeout protection
                try:
                    await asyncio.wait_for(
                        self._session_context.__aexit__(None, None, None), timeout=10.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "MCP session context exit timed out after 10s, forcing cleanup"
                    )
                finally:
                    self._session_context = None
                    self._session = None
        except Exception as e:
            logger.warning(f"Error during MCP session shutdown: {e}")

        self._all_tools.clear()
        self._initialized = False
        logger.success("MCPSessionManager shutdown complete")


# =======================================================================
# Base Agent Manager
# =======================================================================


class BaseAgentManager:
    """
    Base class for thread-safe agent managers.
    Uses shared MCP session for connection pooling.
    """

    _instances: Dict[AgentType, "BaseAgentManager"] = {}
    _locks: Dict[AgentType, asyncio.Lock] = {t: asyncio.Lock() for t in AgentType}

    def __init__(self, config: AgentConfig):
        self._config = config
        self._agent_config: Optional[Dict[str, Any]] = None
        self._initialized = False
        self._concurrency_semaphore: Optional[asyncio.Semaphore] = None

    @classmethod
    async def get_instance(cls, agent_type: AgentType) -> "BaseAgentManager":
        """Get or create the singleton instance for an agent type."""
        if agent_type not in cls._instances:
            async with cls._locks[agent_type]:
                if agent_type not in cls._instances:
                    config = AGENT_CONFIGS[agent_type]
                    cls._instances[agent_type] = cls(config)
        return cls._instances[agent_type]

    @classmethod
    def reset_instance(cls, agent_type: AgentType) -> None:
        """Reset instance for testing purposes."""
        if agent_type in cls._instances:
            del cls._instances[agent_type]

    async def initialize(self, mcp_manager: MCPSessionManager) -> None:
        """
        Initialize the agent configuration using the shared MCP session.
        """
        if self._initialized:
            logger.warning(f"{self._config.name} already initialized, skipping...")
            return

        logger.info(f"Initializing {self._config.name}...")

        # Get tools from shared session
        tools = mcp_manager.get_tools(self._config.tool_names)

        if not tools:
            raise RuntimeError(f"No tools found for {self._config.name}")

        self._agent_config = {
            "name": self._config.name,
            "model_client": agent_az_client,
            "tools": tools,
            "system_message": self._config.system_prompt,
            "reflect_on_tool_use": True,
            # "max_tool_iterations" : 1,
            "model_client_stream": True,
        }

        self._concurrency_semaphore = asyncio.Semaphore(
            self._config.max_concurrent_agents
        )
        self._initialized = True

        logger.success(
            f"{self._config.name} initialized with {len(tools)} tools "
            f"(max concurrent: {self._config.max_concurrent_agents})"
        )

    async def create_agent_instance(
        self,
        model_context: Optional[BufferedChatCompletionContext] = None,
        use_concurrency_limit: bool = True,
        **context_kwargs,
    ) -> AssistantAgent:
        """Create a new agent instance with thread-safe tool adapters."""
        if not self._initialized or self._agent_config is None:
            raise RuntimeError(
                f"{self._config.name} not initialized. Call initialize() during app startup."
            )

        if use_concurrency_limit and self._concurrency_semaphore:
            # CRITICAL FIX: Add timeout to prevent indefinite blocking when concurrency limit reached
            try:
                async with asyncio.timeout(30.0):  # 30 second timeout for semaphore acquisition
                    async with self._concurrency_semaphore:
                        return await self._create_agent_internal(
                            model_context, **context_kwargs
                        )
            except asyncio.TimeoutError:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=503,
                    detail=f"Service overloaded. Max concurrent {self._config.name} agents ({self._config.max_concurrent_agents}) reached. Please try again later."
                )
        else:
            return await self._create_agent_internal(model_context, **context_kwargs)

    async def _create_agent_internal(
        self, model_context: Optional[BufferedChatCompletionContext], **context_kwargs
    ) -> AssistantAgent:
        """Internal method to create agent instance."""
        if model_context is None:
            model_context = BufferedChatCompletionContext(
                buffer_size=CONVERSATION_BUFFER_SIZE
            )

        agent_config = self._agent_config.copy()

        # CRITICAL FIX: Extract and apply custom_system_message if provided
        custom_system_message = context_kwargs.pop("custom_system_message", None)

        if custom_system_message:
            # Override the default system message with the custom one
            agent_config["system_message"] = custom_system_message
            logger.info(f"Applied custom system message for {self._config.role_name}")

        # Inject context if other kwargs are provided
        if context_kwargs and any(v is not None for v in context_kwargs.values()):
            context_injection = build_context_injection(
                self._config.role_name, **context_kwargs
            )
            agent_config["system_message"] = (
                agent_config["system_message"] + context_injection
            )
            logger.info(f"Injected {self._config.role_name} context: {context_kwargs}")

        agent = AssistantAgent(**agent_config, model_context=model_context)
        logger.debug(f"Created new {self._config.name} instance")
        return agent

    def get_agent_config(self) -> Dict[str, Any]:
        """Get the raw agent configuration."""
        if not self._initialized or self._agent_config is None:
            raise RuntimeError(f"{self._config.name} not initialized")
        return self._agent_config

    @property
    def is_initialized(self) -> bool:
        """Check if manager is initialized."""
        return self._initialized

    def get_concurrency_stats(self) -> Dict[str, Any]:
        """Get current concurrency statistics."""
        if not self._concurrency_semaphore:
            return {"enabled": False}

        return {
            "enabled": True,
            "max_concurrent": self._config.max_concurrent_agents,
            "available_slots": self._concurrency_semaphore._value,
            "active_agents": self._config.max_concurrent_agents
            - self._concurrency_semaphore._value,
        }

    async def shutdown(self) -> None:
        """Cleanup resources during application shutdown."""
        logger.info(f"Shutting down {self._config.name}...")
        self._agent_config = None
        self._initialized = False
        logger.success(f"{self._config.name} shutdown complete")


# =======================================================================
# Unified Agent Manager (Facade)
# =======================================================================


class UnifiedAgentManager:
    """
    Unified facade for managing all agent types.
    Uses a shared MCP session for efficient connection management.
    """

    _instance: Optional["UnifiedAgentManager"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._managers: Dict[AgentType, BaseAgentManager] = {}
        self._mcp_manager: Optional[MCPSessionManager] = None
        self._initialized = False

    @classmethod
    async def get_instance(cls) -> "UnifiedAgentManager":
        """Get or create the singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = UnifiedAgentManager()
        return cls._instance

    async def initialize_all(self) -> None:
        """
        Initialize all agent managers with shared MCP session.
        This creates ONE connection to the MCP server for all agents.
        """
        if self._initialized:
            logger.warning("UnifiedAgentManager already initialized")
            return

        logger.info("=" * 60)
        logger.info("Initializing UnifiedAgentManager with shared MCP session...")
        logger.info("=" * 60)

        # Step 1: Initialize shared MCP session (ONE connection)
        self._mcp_manager = await MCPSessionManager.get_instance()
        await self._mcp_manager.initialize()

        # Step 2: Create all manager instances
        for agent_type in AgentType:
            self._managers[agent_type] = await BaseAgentManager.get_instance(agent_type)

        # Step 3: Initialize all managers in parallel (they just filter tools, no new connections)
        await asyncio.gather(
            *[
                manager.initialize(self._mcp_manager)
                for manager in self._managers.values()
            ]
        )

        # Verify all initialized
        for agent_type, manager in self._managers.items():
            if not manager.is_initialized:
                raise RuntimeError(
                    f"Failed to initialize {agent_type.value} agent manager"
                )

        self._initialized = True
        logger.success("=" * 60)
        logger.success("All agent managers initialized with shared MCP session!")
        logger.success("=" * 60)

    async def initialize_specific(self, *agent_types: AgentType) -> None:
        """Initialize specific agent types."""
        if not self._mcp_manager or not self._mcp_manager.is_initialized:
            self._mcp_manager = await MCPSessionManager.get_instance()
            await self._mcp_manager.initialize()

        for agent_type in agent_types:
            if agent_type not in self._managers:
                self._managers[agent_type] = await BaseAgentManager.get_instance(
                    agent_type
                )

        await asyncio.gather(
            *[self._managers[t].initialize(self._mcp_manager) for t in agent_types]
        )

    def get_manager(self, agent_type: AgentType) -> BaseAgentManager:
        """Get a specific agent manager."""
        if agent_type not in self._managers:
            raise RuntimeError(f"{agent_type.value} manager not initialized")
        return self._managers[agent_type]

    @property
    def recruiter(self) -> BaseAgentManager:
        return self.get_manager(AgentType.RECRUITER)

    @property
    def sales(self) -> BaseAgentManager:
        return self.get_manager(AgentType.SALES)

    @property
    def client(self) -> BaseAgentManager:
        return self.get_manager(AgentType.CLIENT)

    @property
    def candidate(self) -> BaseAgentManager:
        return self.get_manager(AgentType.CANDIDATE)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def shutdown_all(self) -> None:
        """
        Shutdown all agent managers and close the shared MCP session.
        This is now FAST because we only close ONE connection.
        """
        logger.info("Shutting down all agent managers...")

        # Shutdown managers first (just clears config, no connection to close)
        await asyncio.gather(
            *[manager.shutdown() for manager in self._managers.values()]
        )

        # Then close the single shared MCP session
        if self._mcp_manager:
            await self._mcp_manager.shutdown()

        self._initialized = False
        logger.success("All agent managers shutdown complete")


# =======================================================================
# Convenience Functions (Backward Compatibility)
# =======================================================================


async def get_unified_manager() -> UnifiedAgentManager:
    """Get the unified agent manager singleton."""
    return await UnifiedAgentManager.get_instance()


async def get_agent_manager() -> BaseAgentManager:
    """Get the recruiter agent manager (legacy compatibility)."""
    return await BaseAgentManager.get_instance(AgentType.RECRUITER)


async def get_sales_agent_manager() -> BaseAgentManager:
    """Get the sales agent manager (legacy compatibility)."""
    return await BaseAgentManager.get_instance(AgentType.SALES)


async def get_client_agent_manager() -> BaseAgentManager:
    """Get the client agent manager (legacy compatibility)."""
    return await BaseAgentManager.get_instance(AgentType.CLIENT)


async def get_candidate_agent_manager() -> BaseAgentManager:
    """Get the candidate agent manager (legacy compatibility)."""
    return await BaseAgentManager.get_instance(AgentType.CANDIDATE)
