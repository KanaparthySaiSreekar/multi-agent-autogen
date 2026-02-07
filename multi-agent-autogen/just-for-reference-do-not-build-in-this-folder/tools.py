# =======================================================================
#                     agents/mcp/tools.py (FIXED COMBINED TOOL)
# =======================================================================

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timezone
from typing import Optional, Dict, Any, List, Union, Callable
from difflib import SequenceMatcher

import json

from fastapi import HTTPException
from fastmcp import FastMCP
from pydantic import ValidationError
from sqlalchemy.future import select

from app.core.logger import logger
from app.db.engine import AsyncSessionLocal
from app.db.models import JobDraft, JobRequirement, Recruiter, Client, JobRequirementAssignment, RecruiterLead, Sales, PipelineDraft, PipelineStageDraft, JobPosting
from app.db.schemas import UserRole
from app.db.config_schemas import TASC_SCHEMA
from sqlalchemy.orm.attributes import flag_modified

# Import schemas
from app.agents.mcp.schemas import (
    DraftJobRequest,
    DraftQuestionsResponse,
    GenerateJDInput,
    GeneratedJD,
    JobRequestParseInput,
    StructuredJobDescription,
)

# Import from jobs module
from app.db.schemas import ProcessType, JobRequirementStatus


from app.jobs.schemas.job_schemas import (
    CreateJobWithDraftInput,
    GetJobPipelineInput,
    GetJobPipelineResponse,
)

# Import from candidates module
from app.candidates.schemas.candidate_schemas import (
    SendPrescreeningInvitesInput,
    SendPrescreeningInvitesResponse,
)

# Import services
from app.agents.mcp.services import (
    JobDescriptionService,
)

from app.chat.candidate_query.schema.candidate_query_schema import (
    RunCandidateQueryInput,
    CandidateSqlQueryResponse,
)
from app.agents.mcp.services import JobParsingService, UserRepository
from app.jobs.services.job_service import JobService
from app.jobs.services.job_draft_service import job_draft_service
from app.jobs.services.pipeline_service import PipelineService
from app.candidates.services.sourcing_service import CandidateSourcingService
from app.analytics.services.analytics_service import AnalyticsService
from app.utils.schema_index import SchemaIndex
from app.chat.candidate_query.services.candidate_chat_service import (
    CandidateQueryService,
)
from app.chat.client_chat.schemas.client_chat_schemas import (
    CreateJobRequirementInput,
    CreateJobRequirementResponse,
) 

from app.jobs.services.job_requirement_service import JobRequirementService
from app.ceipal.services.ceipal_authentication import CEIPAL
from app.config import settings

from app.notifications.schemas.notification_schemas import NotifyPayload
from app.notifications.services.notification_service import (
    run_task_and_notifies_reliably,
)


mcp = FastMCP("HiringAgentTools")

# Configure MCP server logger
mcp_server_logger = logging.getLogger("fastmcp.mcp_server")
mcp_server_logger.setLevel(logging.WARNING)

# Get database schema for analytics tool
DB_SCHEMA_FOR_TOOL = AnalyticsService.get_db_schema_string()


# =======================================================================
#                    Tool Response Validation Utilities
# =======================================================================

from functools import wraps
from pydantic import BaseModel


def ensure_json_response(func: Callable) -> Callable:
    """
    Decorator to ensure tool responses are always valid JSON.

    Handles:
    - Pydantic models: Converts to JSON string using model_dump()
    - Dicts: Converts to JSON string using json.dumps()
    - Strings: Validates JSON and returns as-is
    - Empty/None: Returns error JSON
    - Exceptions: Catches and returns error JSON

    This prevents "Expecting value: line 1 column 1 (char 0)" errors
    when tools return empty or malformed responses.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)

            # Handle None or empty results
            if result is None or result == "":
                logger.error(f"Tool '{func.__name__}' returned empty result")
                return json.dumps({
                    "error": f"Tool '{func.__name__}' returned empty result",
                    "success": False
                }, indent=2)

            # Handle Pydantic models
            if isinstance(result, BaseModel):
                try:
                    return json.dumps(result.model_dump(), indent=2, default=str)
                except Exception as e:
                    logger.error(f"Failed to serialize Pydantic model in '{func.__name__}': {e}")
                    return json.dumps({
                        "error": f"Failed to serialize response: {str(e)}",
                        "success": False
                    }, indent=2)

            # Handle dicts
            if isinstance(result, dict):
                try:
                    return json.dumps(result, indent=2, default=str)
                except Exception as e:
                    logger.error(f"Failed to serialize dict in '{func.__name__}': {e}")
                    return json.dumps({
                        "error": f"Failed to serialize response: {str(e)}",
                        "success": False
                    }, indent=2)

            # Handle strings (validate JSON)
            if isinstance(result, str):
                try:
                    # Try to parse to validate JSON
                    json.loads(result)
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"Tool '{func.__name__}' returned invalid JSON: {e}")
                    # If it's not valid JSON, wrap it as an error
                    return json.dumps({
                        "error": "Tool returned invalid JSON",
                        "raw_response": result[:500],  # First 500 chars
                        "success": False
                    }, indent=2)

            # Unexpected type
            logger.error(f"Tool '{func.__name__}' returned unexpected type: {type(result)}")
            return json.dumps({
                "error": f"Tool returned unexpected type: {type(result).__name__}",
                "success": False
            }, indent=2)

        except HTTPException as he:
            # Convert HTTP exceptions to JSON (don't re-raise to MCP)
            logger.error(f"HTTPException in '{func.__name__}': status={he.status_code}, detail={he.detail}")
            return json.dumps({
                "error": str(he.detail),
                "status_code": he.status_code,
                "success": False
            }, indent=2)
        except Exception as e:
            logger.error(f"Unexpected error in '{func.__name__}': {e}", exc_info=True)
            return json.dumps({
                "error": f"Unexpected error: {str(e)}",
                "success": False
            }, indent=2)

    return wrapper


# =======================================================================
#                    String Parsing Utilities
# =======================================================================


def parse_pipe_separated_input(
    input_data: str, min_parts: int, error_message: str
) -> List[str]:
    """
    Utility function to parse pipe-separated input with validation.

    Args:
        input_data: Pipe-separated string
        min_parts: Minimum number of parts required
        error_message: Error message to return if validation fails

    Returns:
        List of parsed parts

    Raises:
        ValueError: If input doesn't have minimum required parts
    """
    parts = input_data.split("|")
    if len(parts) < min_parts:
        raise ValueError(error_message)
    return [part.strip() for part in parts]


def parse_key_value_pairs(
    pairs_str: str, separator: str = ",", kv_separator: str = ":"
) -> Dict[str, Any]:
    """
    Parse key-value pairs from a string.

    Args:
        pairs_str: String containing key-value pairs
        separator: Separator between pairs (default: ",")
        kv_separator: Separator between key and value (default: ":")

    Returns:
        Dict of parsed key-value pairs

    Example:
        parse_key_value_pairs("key1:value1,key2:value2")
        -> {"key1": "value1", "key2": "value2"}
    """
    result = {}
    if not pairs_str:
        return result

    for pair in pairs_str.split(separator):
        if kv_separator in pair:
            key, value = pair.split(kv_separator, 1)
            result[key.strip()] = value.strip()

    return result


def parse_boolean(value: str) -> bool:
    """
    Parse boolean value from string.

    Args:
        value: String representation of boolean

    Returns:
        Boolean value
    """
    return value.strip().lower() == "true"


async def get_country_currency_from_ceipal(
    country_name: str, db_session
) -> Dict[str, Any]:
    """
    Look up a country in Ceipal and return currency information.
    If no exact match, returns closest matches for the user to select from.

    Args:
        country_name: The country name to search for
        db_session: Database session

    Returns:
        Dict with:
        - found: bool - True if exact match found
        - country_id: int - Currency ID (country ID) if found
        - country_name: str - Matched country name if found
        - suggestions: List[Dict] - List of closest matches if not found
        - error: str - Error message if validation failed

    Example response (exact match):
        {
            "found": True,
            "country_id": 237,
            "country_name": "United States",
            "currency": "Dollar",
            "iso_currency_code": "USD"
        }

    Example response (no exact match):
        {
            "found": False,
            "suggestions": [
                {"id": 237, "name": "United States", "similarity": 0.85},
                {"id": 238, "name": "United States Minor Outlying Islands", "similarity": 0.75},
                {"id": 232, "name": "U.S. Virgin Islands", "similarity": 0.60}
            ],
            "error": "No exact match found for '[country_name]'. Please select from suggestions."
        }
    """

    try:
        # Initialize CEIPAL client
        ceipal = CEIPAL(
            email=settings.ceipal_email,
            password=settings.ceipal_password,
            api_key=settings.ceipal_api_key,
            db=db_session,
        )

        # Try exact match lookup
        lookup_result = await ceipal.lookup_country_in_api(country_name=country_name.lower())

        if lookup_result.get("found") and lookup_result.get("country_data"):
            country_data = lookup_result["country_data"]
            return {
                "found": True,
                "country_id": country_data.get("id"),
                "country_name": country_data.get("name"),
                "currency": country_data.get("currency"),
                "iso_currency_code": country_data.get("iso_currency_code"),
            }

        # No exact match - get all countries and find closest matches
        api_url = f"https://{settings.CEIPAL_START_URL}/{settings.ceipal_api_key}/countriesList/"
        token = await ceipal.get_valid_access_token()

        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                api_url,
                headers={"Authorization": f"Bearer {token}"},
                json={},
                timeout=30.0,
            )
            response.raise_for_status()
            countries_data = response.json()

        if not countries_data or "data" not in countries_data:
            return {
                "found": False,
                "error": "Unable to fetch countries list from Ceipal API."
            }

        countries = countries_data["data"]

        # Calculate similarity scores for all countries
        def similarity(a: str, b: str) -> float:
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        country_matches = []
        for country in countries:
            country_name_api = country.get("name", "")
            score = similarity(country_name, country_name_api)

            # Also check aliases/common names
            if country_name.lower() in ["usa", "us"]:
                if country_name_api.lower() == "united states":
                    score = 1.0
            elif country_name.lower() in ["uk"]:
                if country_name_api.lower() == "united kingdom":
                    score = 1.0
            elif country_name.lower() in ["uae"]:
                if country_name_api.lower() == "united arab emirates":
                    score = 1.0

            if score > 0.3:  # Only include if similarity > 30%
                country_matches.append({
                    "id": country.get("id"),
                    "name": country_name_api,
                    "currency": country.get("currency"),
                    "iso_currency_code": country.get("iso_currency_code"),
                    "similarity": score
                })

        # Sort by similarity and get top 5
        country_matches.sort(key=lambda x: x["similarity"], reverse=True)
        top_matches = country_matches[:5]

        if not top_matches:
            return {
                "found": False,
                "error": f"No matching countries found for '{country_name}'. Please provide a valid country name."
            }

        # Format suggestions for user
        suggestions = [
            {
                "id": match["id"],
                "name": match["name"],
                "currency": match.get("currency"),
                "similarity": round(match["similarity"] * 100, 1)
            }
            for match in top_matches
        ]

        return {
            "found": False,
            "suggestions": suggestions,
            "error": f"No exact match found for '{country_name}'. Did you mean one of these countries?"
        }

    except Exception as e:
        logger.error(f"Error looking up country in Ceipal: {e}")
        return {
            "found": False,
            "error": f"Error validating country: {str(e)}"
        }


def extract_recruiter_id_from_context() -> Optional[uuid.UUID]:
    """Extract recruiter_id from the current execution context.

    This function attempts to extract the recruiter_id from the system message
    context that was injected by the chat service when user context was added.
    """
    import re
    import sys

    try:
        # Look for the recruiter ID in the call stack
        # The system message context should contain "Recruiter ID: {uuid}"
        frame = sys._getframe()
        while frame:
            # Check if we can find the recruiter_id in local variables or context
            if "system_message" in frame.f_locals:
                system_message = frame.f_locals["system_message"]
                if isinstance(system_message, str):
                    match = re.search(
                        r"Recruiter ID:\s*([a-f0-9-]{36})", system_message
                    )
                    if match:
                        try:
                            return uuid.UUID(match.group(1))
                        except ValueError:
                            pass
            frame = frame.f_back

        # If not found in stack, try to get from the current thread or environment
        # This is a fallback approach
        return None

    except Exception as e:
        logger.warning(f"Could not extract recruiter_id from context: {e}")
        return None


def extract_sales_id_from_context() -> Optional[uuid.UUID]:
    """Extract sales_id from the current execution context.

    This function attempts to extract the sales_id from the system message
    context that was injected by the sales agent manager when user context was added.
    """
    import re
    import sys

    try:
        # Look for the sales ID in the call stack
        # The system message context should contain "Sales ID: {uuid}"
        frame = sys._getframe()
        while frame:
            # Check if we can find the sales_id in local variables or context
            if "system_message" in frame.f_locals:
                system_message = frame.f_locals["system_message"]
                if isinstance(system_message, str):
                    match = re.search(r"Sales ID:\s*([a-f0-9-]{36})", system_message)
                    if match:
                        try:
                            return uuid.UUID(match.group(1))
                        except ValueError:
                            pass
            frame = frame.f_back

        # If not found in stack, try to get from the current thread or environment
        # This is a fallback approach
        return None

    except Exception as e:
        logger.warning(f"Could not extract sales_id from context: {e}")
        return None


# =======================================================================
#                       FIXED COMBINED TOOL
# =======================================================================


@mcp.tool(
    description="Parse a job request AND generate a complete job description with pre-screening questions in one step. Provide job_requirement_id (to link to an existing requirement), location, experience, skills, number_of_positions, time_to_fill_days, and min_pay_rate (minimum pay rate amount). The currency will be automatically inferred from the country in the location. Pipeline will use default sourcing (50 candidates) and ranking (top 25) configs - use configure-sourcing/configure-ranking routes to customize."
)
@ensure_json_response
async def create_job_from_text(input_data: JobRequestParseInput) -> Union[GeneratedJD, str]:
    """FIXED: Combined tool that properly handles skills and infers currency from country"""
    logger.info("Tool 'create_job_from_text' called - parsing and generating JD.")

    try:
        from app.agents.mcp.services import JobParsingService

        parsed_result = await JobParsingService.parse_job_request_detailed(input_data)
        logger.info(
            f"Parsed job: {parsed_result.title} with {len(parsed_result.skills)} detailed skills"
        )

        # If pay_rate_currency is not provided, infer from location_country
        pay_rate_currency = input_data.pay_rate_currency
        if not pay_rate_currency and parsed_result.location_country:
            async with AsyncSessionLocal() as validation_session:
                country_validation = await get_country_currency_from_ceipal(
                    parsed_result.location_country, validation_session
                )

            if not country_validation.get("found"):
                # No exact match - return error with suggestions
                suggestions = country_validation.get("suggestions", [])
                if suggestions:
                    suggestion_text = "\n".join([
                        f"{i+1}. {s['name']} (Currency: {s.get('currency', 'N/A')})"
                        for i, s in enumerate(suggestions)
                    ])
                    error_msg = (
                        f"{country_validation.get('error', 'Country not found')}\n\n"
                        f"Please select one of these countries:\n{suggestion_text}\n\n"
                        f"Provide the full country name from the list above."
                    )
                else:
                    error_msg = country_validation.get("error", f"Country '{parsed_result.location_country}' not found in Ceipal countries list.")

                raise HTTPException(status_code=400, detail=error_msg)

            pay_rate_currency = country_validation["country_id"]
            logger.info(f"Inferred currency ID {pay_rate_currency} from country '{country_validation['country_name']}'")

        jd_input = GenerateJDInput(
            title=parsed_result.title,
            skills=parsed_result.skills,
            location=parsed_result.location,
            location_country=parsed_result.location_country,
            experience=parsed_result.experience,
            max_experience=parsed_result.max_experience,
            domain=parsed_result.domain,
            number_of_positions=parsed_result.number_of_positions,
            time_to_fill_days=parsed_result.time_to_fill_days,
            priority=parsed_result.priority,
            pay_rate_currency=pay_rate_currency,
            min_pay_rate=input_data.min_pay_rate,
            job_requirement_id=input_data.job_requirement_id,
        )

        logger.info(f"JD Input skills: {jd_input.skills}")

        async with AsyncSessionLocal() as session:
            async with session.begin():
                result = await JobDescriptionService.generate_jd(
                    db=session, input_data=jd_input
                )
                logger.success("Successfully created job from text in one step.")
                return result

    except HTTPException as he:
        # Re-raise HTTP exceptions
        logger.error(f"HTTP error in create_job_from_text: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error in create_job_from_text: {e}")
        raise HTTPException(status_code=500, detail=f"Combined job creation error: {e}")


@mcp.tool(
    description=(
        "Create a job draft directly from an existing job requirement without re-generating the JD. "
        "Provide the job requirement UUID as a string, min_pay_rate (minimum pay rate amount), and location_country. "
        "IMPORTANT: You MUST ask the user for the country and provide it via the location_country parameter. "
        "The currency will be automatically determined from the country. "
        "If the country doesn't match exactly, the tool will return suggestions for you to present to the user."
    )
)
@ensure_json_response
async def create_job_draft_from_requirement(
    job_requirement_id: str,
    min_pay_rate: float,
    location_country: str
) -> Union[GeneratedJD, str]:
    """
    Build a fresh job draft by copying the structured job description and metadata
    from an existing job requirement instead of calling the LLM again.

    Args:
        job_requirement_id: UUID of the job requirement to create draft from
        min_pay_rate: Minimum pay rate amount
        location_country: Country for the job location (required, will be validated and used to infer currency)
    """
    logger.info(
        "Tool 'create_job_draft_from_requirement' called for requirement: %s",
        job_requirement_id,
    )

    if not job_requirement_id or not isinstance(job_requirement_id, str):
        raise HTTPException(
            status_code=400,
            detail="job_requirement_id must be provided as a UUID string",
        )

    try:
        requirement_uuid = uuid.UUID(job_requirement_id.strip())
    except ValueError as exc:
        logger.error("Invalid job_requirement_id format: %s", exc)
        raise HTTPException(
            status_code=400, detail="Invalid job_requirement_id format"
        ) from exc

    try:
        async with AsyncSessionLocal() as session:
            job_req_stmt = select(JobRequirement).where(
                JobRequirement.job_requirement_id == requirement_uuid
            )
            job_req_result = await session.execute(job_req_stmt)
            job_req = job_req_result.scalar_one_or_none()

            if not job_req:
                raise HTTPException(
                    status_code=404,
                    detail=f"Job requirement {job_requirement_id} not found",
                )

            existing_draft_stmt = (
                select(JobDraft)
                .where(
                    JobDraft.job_requirement_id == requirement_uuid,
                    JobDraft.is_active == True,  # noqa: E712 - SQLAlchemy comparison
                    JobDraft.expires_at > datetime.utcnow(),
                )
                .order_by(JobDraft.created_at.desc())
            )
            existing_draft_result = await session.execute(existing_draft_stmt)
            existing_draft = existing_draft_result.scalars().first()

        if existing_draft:
            logger.info(
                "Found active draft %s linked to requirement %s; reusing it.",
                existing_draft.draft_id,
                job_requirement_id,
            )
            structured_source = existing_draft.structured_jd or job_req.structured_jd
            if not structured_source:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Existing draft is missing structured JD data. "
                        "Please regenerate the job description."
                    ),
                )
            try:
                structured_jd = StructuredJobDescription.model_validate(
                    structured_source
                )
            except ValidationError as exc:
                logger.error(
                    "Stored structured JD on draft %s is invalid: %s",
                    existing_draft.draft_id,
                    exc,
                )
                raise HTTPException(
                    status_code=400,
                    detail="Stored draft JD is invalid; please regenerate the job description.",
                ) from exc

            return GeneratedJD(
                title=existing_draft.title,
                structured_jd=structured_jd,
                full_text=existing_draft.full_text,
                process_id=existing_draft.draft_id,
            )

        structured_source = job_req.structured_jd or job_req.job_requirement_details
        if not structured_source:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Job requirement is missing structured JD data. "
                    "Please update the job requirement with a job description first."
                ),
            )

        try:
            structured_jd = StructuredJobDescription.model_validate(structured_source)
        except ValidationError as exc:
            logger.error(
                "Job requirement %s has invalid structured JD: %s",
                job_requirement_id,
                exc,
            )
            raise HTTPException(
                status_code=400,
                detail="Job requirement structured_jd is invalid; please fix the data and try again.",
            ) from exc

        full_text = job_req.job_description_full_text
        if not full_text:
            full_text = JobDescriptionService._format_jd_to_text(
                job_req.title, structured_jd
            )

        raw_skills = job_req.skills or []
        skills_for_draft: List[Dict[str, Any]] = []
        for skill in raw_skills:
            if isinstance(skill, dict):
                name = (
                    skill.get("name")
                    or skill.get("skill_name")
                    or skill.get("SkillName")
                )
                if not name:
                    continue
                skill_type = skill.get("skill_type") or skill.get("type") or "technical"
                skill_type = str(skill_type).strip().lower()
                if skill_type not in {"technical", "non-technical"}:
                    logger.warning(
                        "Skill '%s' has unsupported skill_type '%s'; defaulting to 'technical'",
                        name,
                        skill_type,
                    )
                    skill_type = "technical"

                experience_years = skill.get("experience_years")
                if experience_years is not None:
                    try:
                        experience_years = int(experience_years)
                        if experience_years < 0:
                            experience_years = None
                    except (TypeError, ValueError):
                        logger.warning(
                            "Could not parse experience years '%s' for skill '%s'; ignoring.",
                            experience_years,
                            name,
                        )
                        experience_years = None

                skills_for_draft.append(
                    {
                        "name": str(name).strip(),
                        "skill_type": skill_type,
                        "experience_years": experience_years,
                    }
                )
            elif isinstance(skill, str):
                skills_for_draft.append(
                    {
                        "name": skill.strip(),
                        "skill_type": "technical",
                        "experience_years": None,
                    }
                )

        priority_value = None
        if job_req.priority:
            priority_value = (
                job_req.priority.value
                if hasattr(job_req.priority, "value")
                else str(job_req.priority).upper()
            )
            logger.info(f"✓ Extracted priority from job requirement: {priority_value}")

        if priority_value not in {"HIGH", "MEDIUM", "LOW"}:
            error_msg = f"❌ Priority value '{priority_value}' is invalid. Must be HIGH, MEDIUM, or LOW."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"✓ Using priority value: {priority_value}")

        number_of_positions = job_req.openings or 1
        time_to_fill_days = job_req.sla_days
        domain_value = getattr(job_req, "domain", None)

        # Validate country and get currency ID
        async with AsyncSessionLocal() as validation_session:
            country_validation = await get_country_currency_from_ceipal(location_country, validation_session)

        if not country_validation.get("found"):
            # No exact match - return error with suggestions
            suggestions = country_validation.get("suggestions", [])
            if suggestions:
                suggestion_text = "\n".join([
                    f"{i+1}. {s['name']} (Currency: {s.get('currency', 'N/A')})"
                    for i, s in enumerate(suggestions)
                ])
                error_msg = (
                    f"{country_validation.get('error', 'Country not found')}\n\n"
                    f"Please select one of these countries:\n{suggestion_text}\n\n"
                    f"Provide the full country name from the list above."
                )
            else:
                error_msg = country_validation.get("error", f"Country '{location_country}' not found in Ceipal countries list.")

            raise HTTPException(status_code=400, detail=error_msg)

        # Extract currency ID from validated country
        pay_rate_currency = country_validation["country_id"]
        parsed_location_country = country_validation["country_name"]
        parsed_location_city = job_req.location

        # Try to parse city from location if it contains country
        if job_req.location and "," in job_req.location:
            parts = job_req.location.split(",", 1)
            parsed_location_city = parts[0].strip()
            logger.info(f"Parsed location into city: '{parsed_location_city}', country: '{parsed_location_country}'")

        logger.info(f"Inferred currency ID {pay_rate_currency} from country '{parsed_location_country}'")

        draft_result = await job_draft_service.create_draft(
            title=job_req.title,
            structured_jd=structured_jd.model_dump(),
            full_text=full_text,
            skills=skills_for_draft,
            location=parsed_location_city,
            location_country=parsed_location_country,
            experience=job_req.experience,
            max_experience=job_req.max_experience,
            domain=domain_value,
            number_of_positions=number_of_positions,
            time_to_fill_days=time_to_fill_days,
            priority=priority_value,
            pay_rate_currency=pay_rate_currency,
            min_pay_rate=min_pay_rate,
            job_requirement_id=requirement_uuid,
        )

        # Extract draft_id from result
        draft_id_str = draft_result["draft_id"]
        draft_uuid = uuid.UUID(draft_id_str)

        # Log country validation message if present
        if draft_result.get("country_validation_message"):
            logger.info(f"Country validation: {draft_result['country_validation_message']}")

        # Generate prescreening questions synchronously
        try:
            await JobDescriptionService._background_generate_questions(
                draft_id_str, full_text
            )
            logger.info(f"Successfully generated prescreening questions for draft {draft_id_str}")
        except Exception as e:
            logger.error(f"Failed to generate questions for draft {draft_id_str}: {e}")
            # Draft still exists, user can add questions manually later

        logger.success(
            "Created job draft %s from job requirement %s using existing data.",
            draft_uuid,
            job_requirement_id,
        )

        return GeneratedJD(
            title=job_req.title,
            structured_jd=structured_jd,
            full_text=full_text,
            process_id=draft_uuid,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error in create_job_draft_from_requirement: %s",
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create draft from requirement: {exc}",
        ) from exc


@mcp.tool(
    description=(
        "Edit a job draft and optionally its linked job requirement. "
        "Supports partial updates - only provide fields you want to change. "
        "Format: draft_id|updates_json\n\n"
        "IMPORTANT: After editing, the JD and prescreening questions are automatically regenerated. "
        "If a pipeline exists, the PRESCREENING stage questions are also updated.\n\n"
        "Editable fields:\n"
        "- title: str\n"
        "- location: str (city)\n"
        "- location_country: str (triggers CEIPAL validation and currency update)\n"
        "- experience: int (minimum years)\n"
        "- max_experience: int (maximum years)\n"
        "- domain: str\n"
        "- number_of_positions: int\n"
        "- time_to_fill_days: int\n"
        "- priority: str (HIGH/MEDIUM/LOW)\n"
        "- employment_type: str (PERMANENT/CONTRACT/REMOTE)\n"
        "- min_pay_rate: float\n"
        "- skills: List of skills - supports smart merge:\n"
        "  - add_skills: [{name, skill_type, experience_years}] - adds new skills\n"
        "  - remove_skills: [skill_names] - removes skills by name\n"
        "  - replace_skills: [{name, skill_type, experience_years}] - replaces all skills\n"
        "- questions: List[{question: str}] - prescreening questions\n"
        "- full_text: str - job description markdown\n\n"
        "Example: draft_id|{\"title\": \"Senior Python Dev\", \"add_skills\": [{\"name\": \"Django\", \"skill_type\": \"technical\", \"experience_years\": 3}]}\n"
        "If draft is linked to a job requirement, both will be updated."
    )
)
@ensure_json_response
async def edit_job_draft(input_data: str) -> str:
    """
    Edit a job draft with partial updates and smart skills merge.

    Supports:
    - Partial field updates (PATCH style)
    - Smart skills merge (add/remove/replace)
    - Country validation with CEIPAL
    - Draft + requirement sync

    Args:
        input_data: Pipe-separated string "draft_id|updates_json"

    Returns:
        JSON string with success/error status and updated fields
    """
    logger.info(f"Tool 'edit_job_draft' called with input: {input_data[:100]}...")

    try:
        # Parse pipe-separated input
        if "|" not in input_data:
            return json.dumps({
                "success": False,
                "error": "Invalid input format. Expected: draft_id|updates_json"
            })

        draft_id_str, updates_json = input_data.split("|", 1)
        draft_id_str = draft_id_str.strip()

        # Parse updates JSON
        try:
            updates = json.loads(updates_json)
        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"Invalid JSON in updates: {str(e)}"
            })

        if not isinstance(updates, dict):
            return json.dumps({
                "success": False,
                "error": "Updates must be a JSON object (dictionary)"
            })

        # Validate UUID
        try:
            draft_uuid = uuid.UUID(draft_id_str)
        except ValueError:
            return json.dumps({
                "success": False,
                "error": f"Invalid draft_id UUID: {draft_id_str}"
            })

        logger.info(f"Editing draft {draft_uuid} with updates: {updates}")

        # Update the draft using the comprehensive service method
        async with AsyncSessionLocal() as session:
            from app.jobs.services.job_draft_service import JobDraftService

            result = await JobDraftService.update_draft_comprehensive(
                draft_id=draft_id_str,
                updates=updates,
                validate_country=True
            )

            # If update failed, return error
            if not result.get("success"):
                return json.dumps(result)

            # Fetch the updated draft to get job_requirement_id
            draft_result = await session.execute(
                select(JobDraft).where(JobDraft.draft_id == draft_uuid)
            )
            draft = draft_result.scalar_one_or_none()

            if not draft:
                return json.dumps({
                    "success": False,
                    "error": "Draft not found after update"
                })

            # If draft is linked to a job requirement, sync updates
            requirement_updated = False
            if draft.job_requirement_id:
                logger.info(f"Syncing updates to linked job requirement {draft.job_requirement_id}")

                # Map draft fields to requirement fields
                req_updates = {}

                # Direct field mappings
                if "title" in updates:
                    req_updates["title"] = updates["title"]

                if "location" in updates or "location_country" in updates:
                    # Requirement stores location as "city, country"
                    city = updates.get("location", draft.location)
                    country = updates.get("location_country", draft.location_country)
                    if city and country:
                        req_updates["location"] = f"{city}, {country}"
                    elif country:
                        req_updates["location"] = country

                if "experience" in updates:
                    req_updates["experience"] = updates["experience"]

                if "max_experience" in updates:
                    req_updates["max_experience"] = updates["max_experience"]

                if "domain" in updates:
                    req_updates["domain"] = updates["domain"]

                if "priority" in updates:
                    req_updates["priority"] = updates["priority"]

                if "employment_type" in updates:
                    req_updates["employment_type"] = updates["employment_type"]

                # Handle skills - need to convert format
                # Draft uses "name", requirement uses "skill_name"
                final_skills = result.get("skills")
                if final_skills:
                    req_skills = [
                        {
                            "skill_name": s["name"],
                            "skill_type": s["skill_type"],
                            "experience_years": s.get("experience_years")
                        }
                        for s in final_skills
                    ]
                    req_updates["skills"] = req_skills

                # Update the requirement if we have any updates
                if req_updates:
                    try:
                        await JobRequirementService.update_job_requirement(
                            db=session,
                            job_requirement_id=str(draft.job_requirement_id),
                            updates=req_updates
                        )
                        requirement_updated = True
                        logger.info(f"Successfully synced updates to job requirement {draft.job_requirement_id}")
                    except Exception as req_error:
                        logger.error(f"Failed to update linked job requirement: {req_error}")
                        # Don't fail the entire operation if requirement update fails
                        # The draft was updated successfully
                        result["requirement_update_warning"] = f"Draft updated but requirement sync failed: {str(req_error)}"

            # Add requirement sync status to result
            result["requirement_updated"] = requirement_updated

            # Update success message
            if requirement_updated:
                result["message"] = "Successfully updated draft and linked requirement"

            # Regenerate JD and questions after updates
            logger.info("Regenerating JD and prescreening questions after edit...")
            try:
                # Prepare input for JD generation
                jd_input = GenerateJDInput(
                    title=draft.title,
                    skills=draft.skills or [],
                    location=draft.location,
                    location_country=draft.location_country,
                    experience=draft.experience,
                    max_experience=draft.max_experience,
                    domain=draft.domain,
                    number_of_positions=draft.number_of_positions,
                    time_to_fill_days=draft.time_to_fill_days,
                    priority=draft.priority,
                    pay_rate_currency=draft.pay_info.get("pay_rate_currency") if draft.pay_info else None,
                    min_pay_rate=draft.pay_info.get("min_pay_rate") if draft.pay_info else None,
                    job_requirement_id=str(draft.job_requirement_id) if draft.job_requirement_id else None,
                )

                # Generate new JD
                structured_jd, full_text_jd = await JobDescriptionService.generate_jd_for_requirement(jd_input)

                # Update draft with new JD (convert Pydantic model to dict)
                draft.structured_jd = structured_jd.model_dump() if hasattr(structured_jd, 'model_dump') else structured_jd
                draft.full_text = full_text_jd
                flag_modified(draft, "structured_jd")
                await session.commit()
                logger.info("Successfully regenerated JD")

                # Generate new prescreening questions
                await JobDescriptionService._background_generate_questions(
                    str(draft.draft_id), full_text_jd
                )
                logger.info("Successfully regenerated prescreening questions")

                # Update pipeline stage questions if pipeline exists
                pipeline_stmt = select(PipelineDraft).where(
                    PipelineDraft.job_draft_id == draft.draft_id
                )
                pipeline_result = await session.execute(pipeline_stmt)
                pipeline_draft = pipeline_result.scalar_one_or_none()

                if pipeline_draft:
                    logger.info(f"Found pipeline for draft, updating PRESCREENING stage questions...")

                    # Get PRESCREENING stage
                    stage_stmt = select(PipelineStageDraft).where(
                        PipelineStageDraft.pipeline_draft_id == pipeline_draft.pipeline_draft_id,
                        PipelineStageDraft.process_type == ProcessType.PRESCREENING
                    )
                    stage_result = await session.execute(stage_stmt)
                    prescreening_stage = stage_result.scalar_one_or_none()

                    if prescreening_stage:
                        # Reload draft to get updated questions
                        await session.refresh(draft)
                        if draft.questions:
                            prescreening_stage.questions = draft.questions
                            flag_modified(prescreening_stage, "questions")
                            await session.commit()
                            logger.info(f"Updated {len(draft.questions)} questions in PRESCREENING pipeline stage")
                            result["pipeline_questions_updated"] = True
                        else:
                            logger.warning("No questions generated for draft")
                    else:
                        logger.info("No PRESCREENING stage found in pipeline")
                else:
                    logger.info("No pipeline found for draft")

                result["jd_regenerated"] = True
                result["questions_regenerated"] = True

            except Exception as regen_error:
                logger.error(f"Failed to regenerate JD/questions: {regen_error}")
                result["regeneration_warning"] = f"Draft updated but JD/questions regeneration failed: {str(regen_error)}"

            logger.success(f"Successfully edited draft {draft_uuid}")
            return json.dumps(result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unexpected error in edit_job_draft: {exc}")
        return json.dumps({
            "success": False,
            "error": f"Failed to edit draft: {str(exc)}"
        })


@mcp.tool(
    description="Creates a job posting from a reviewed and finalized draft. Use this after the user has reviewed the JD and questions. If recruiter_id is provided, the job will be assigned to that recruiter."
)
@ensure_json_response
async def create_job_posting_from_draft(
    input_data: CreateJobWithDraftInput,
) -> str:
    """Create job posting from finalized draft"""
    logger.info(
        f"Tool 'create_job_posting_from_draft' called for draft: {input_data.draft_id}"
    )

    # Extract recruiter_id from input or use None for default
    recruiter_id = getattr(input_data, "recruiter_id", None)

    try:
        async with AsyncSessionLocal() as session:

            user_ids = await UserRepository.get_all_related_users_for_recruiter(session, recruiter_id)

            logger.info(f"Got the users: {user_ids}")

            ROLE_MESSAGE_MAP = {
                UserRole.RECRUITER: "You have successfully posted a job.",
                UserRole.RECRUITER_LEAD: "A recruiter under you has successfully posted a job.",
                UserRole.SALES: "A recruiter under your sales lead has successfully posted a job.",
                UserRole.RECRUITER_ADMIN: "A recruiter has successfully posted a job within your admin group.",
                UserRole.CLIENT_USER: "A recruiter associated with your account has posted a new job.",
            }

            notifications = [
                NotifyPayload(
                    user_id=str(row["user_id"]),
                    event={"message": ROLE_MESSAGE_MAP.get(
                        row["role"], "Recruiter reassignment update."
                    )},
                )
                for row in user_ids
            ]


            result = await run_task_and_notifies_reliably(
                JobService.create_job_posting_from_draft,
                notifications,
                db=session,
                draft_id=input_data.draft_id,
                recruiter_id=recruiter_id,
            )

        response_payload = result.model_dump()
        logger.success(
            f"create_job_posting_from_draft succeeded for draft {input_data.draft_id}"
        )
        return json.dumps(response_payload, default=str)

    except HTTPException as he:
        logger.error(
            "HTTP error in create_job_posting_from_draft: "
            f"status={he.status_code} detail={he.detail}"
        )
        return json.dumps(
            {
                "error": he.detail,
                "status_code": he.status_code,
                "draft_id": input_data.draft_id,
            }
        )
    except Exception as e:
        logger.exception(
            f"Unexpected error in create_job_posting_from_draft for draft {input_data.draft_id}: {e}"
        )
        return json.dumps(
            {
                "error": f"Unexpected error creating job posting: {str(e)}",
                "draft_id": input_data.draft_id,
            }
        )


@mcp.tool(
    description="Retrieves the pre-screening questions for a job draft so the user can review them."
)
@ensure_json_response
async def get_draft_questions(input_data: DraftJobRequest) -> Union[DraftQuestionsResponse, str]:
    """Get questions from job draft for review"""
    logger.info(f"Tool 'get_draft_questions' called for draft: {input_data.draft_id}")

    draft = await job_draft_service.get_draft(input_data.draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found or expired")

    return DraftQuestionsResponse(
        draft_id=str(draft.draft_id),
        job_title=draft.title,
        questions=draft.questions,
        questions_count=len(draft.questions),
    )


@mcp.tool(
    description="Gets or creates the pipeline configuration for a job DRAFT. Checks if pipeline already exists first - if yes, returns existing config. If no, creates default 6-stage pipeline. Use format: draft_id"
)
@ensure_json_response
async def create_default_pipeline_for_draft(draft_id: str) -> str:
    """Get or create default pipeline configuration for a job draft (smart - doesn't overwrite existing)"""
    logger.info(
        f"Tool 'create_default_pipeline_for_draft' called for draft: {draft_id}"
    )

    try:
        async with AsyncSessionLocal() as session:
            # First, check if pipeline already exists
            try:
                existing_pipeline = (
                    await PipelineService.get_pipeline_draft_for_job_draft(
                        db=session, job_draft_id=uuid.UUID(draft_id)
                    )
                )

                # Pipeline already exists - return it
                if existing_pipeline and existing_pipeline.get("stages"):
                    stage_count = len(existing_pipeline["stages"])
                    active_count = len(
                        [s for s in existing_pipeline["stages"] if s.get("is_active")]
                    )

                    stages_summary = ", ".join(
                        [
                            f"{s['process_type']} (order: {s['process_order']}, active: {s['is_active']})"
                            for s in sorted(
                                existing_pipeline["stages"],
                                key=lambda x: x.get("process_order", 999),
                            )
                            if s.get("is_active")
                        ]
                    )

                    # Return JSON with full pipeline data
                    return json.dumps(
                        {
                            **existing_pipeline,
                            "message": f"Pipeline already exists for this job draft! Total stages: {stage_count}, Active: {active_count}. Active stages: {stages_summary}",
                            "already_exists": True,
                        },
                        indent=2,
                        default=str,
                    )

            except HTTPException as e:
                if e.status_code == 404:
                    pass
                else:
                    raise

            # No pipeline exists - create default one

            # Get default template
            template = await PipelineService.get_default_pipeline_template()

            result = await PipelineService.create_pipeline_draft_for_job_draft(
                db=session, job_draft_id=uuid.UUID(draft_id), pipeline_template=template
            )

            # Return JSON with full pipeline data
            return json.dumps(
                {
                    **result,
                    "message": f"Successfully created default pipeline draft with {len(result['stages'])} stages for job draft {draft_id}. Pipeline includes prescreening questions from the draft.",
                    "already_exists": False,
                },
                indent=2,
                default=str,
            )

    except Exception as e:
        logger.error(f"Error in create_default_pipeline_for_draft: {e}")
        return json.dumps(
            {"error": f"Error creating pipeline draft: {str(e)}"}, indent=2
        )


@mcp.tool(
    description="Configure the sourcing stage for a job draft - sets how many candidates to source from internal database. Use format: draft_id, candidate_limit (1-500)"
)
async def configure_sourcing_stage(draft_id: str, candidate_limit: int) -> str:
    """
    Configure sourcing stage settings.

    Args:
        draft_id: Job draft UUID
        candidate_limit: Number of candidates to source (1-500)

    Returns:
        Success message with configuration details
    """
    logger.info(
        f"Tool 'configure_sourcing_stage' called for draft: {draft_id}, limit: {candidate_limit}"
    )

    try:
        async with AsyncSessionLocal() as session:
            # Validate inputs
            try:
                draft_uuid = uuid.UUID(draft_id)
            except ValueError:
                return json.dumps({"error": f"Invalid draft_id format: {draft_id}"})

            if not (1 <= candidate_limit <= 500):
                return json.dumps(
                    {"error": f"candidate_limit must be between 1-500, got {candidate_limit}"}
                )

            # Get pipeline draft
            stmt = select(PipelineDraft).where(PipelineDraft.job_draft_id == draft_uuid)
            result = await session.execute(stmt)
            pipeline_draft = result.scalar_one_or_none()

            if not pipeline_draft:
                return json.dumps({"error": f"No pipeline found for draft {draft_id}"})

            # Get SOURCING stage
            stmt = (
                select(PipelineStageDraft)
                .where(
                    PipelineStageDraft.pipeline_draft_id
                    == pipeline_draft.pipeline_draft_id,
                    PipelineStageDraft.process_type == ProcessType.SOURCING,
                )
            )
            result = await session.execute(stmt)
            sourcing_stage = result.scalar_one_or_none()

            if not sourcing_stage:
                return json.dumps({"error": "SOURCING stage not found in pipeline"})

            # Update config (APPEND, don't replace)
            if sourcing_stage.process_config is None:
                sourcing_stage.process_config = {}

            sourcing_stage.process_config["candidate_limit"] = candidate_limit
            flag_modified(sourcing_stage, "process_config")

            await session.commit()

            logger.success(
                f"Configured sourcing stage with candidate_limit={candidate_limit} for draft {draft_id}"
            )

            return json.dumps(
                {
                    "success": True,
                    "message": f"Sourcing stage configured to source {candidate_limit} candidates",
                    "stage_draft_id": str(sourcing_stage.stage_draft_id),
                    "config": {"candidate_limit": candidate_limit},
                },
                indent=2,
            )

    except Exception as e:
        logger.error(f"Error in configure_sourcing_stage: {e}")
        return json.dumps({"error": f"Error configuring sourcing stage: {str(e)}"}, indent=2)


@mcp.tool(
    description="Gets the pipeline configuration for a job DRAFT so the user can review it before posting the job. Use format: draft_id"
)
@ensure_json_response
async def get_draft_pipeline(draft_id: str) -> str:
    """Get pipeline configuration from job draft for review"""
    logger.info(f"Tool 'get_draft_pipeline' called for draft: {draft_id}")

    try:
        async with AsyncSessionLocal() as session:
            result = await PipelineService.get_pipeline_draft_for_job_draft(
                db=session, job_draft_id=uuid.UUID(draft_id)
            )

        return json.dumps(result, indent=2, default=str)

    except HTTPException as e:
        if e.status_code == 404:
            return '{"error": "No pipeline draft found for this job draft. Use create_default_pipeline_for_draft first."}'
        raise
    except Exception as e:
        logger.error(f"Error in get_draft_pipeline: {e}")
        return f'{{"error": "{str(e)}"}}'


@mcp.tool(
    description="Updates prescreening questions for a pipeline stage draft. Use this when the user wants to modify questions before posting the job. Use format: stage_draft_id|question1|question2|question3"
)
@ensure_json_response
async def update_draft_pipeline_stage_questions(input_data: str) -> str:
    """Update questions for a pipeline stage draft"""
    logger.info(
        f"Tool 'update_draft_pipeline_stage_questions' called with: {input_data}"
    )

    try:
        parts = input_data.split("|")
        if len(parts) < 2:
            return "Error: Invalid format. Use: stage_draft_id|question1|question2|..."

        stage_draft_id = parts[0]
        questions = [{"question": q.strip()} for q in parts[1:] if q.strip()]

        async with AsyncSessionLocal() as session:
            await PipelineService.update_pipeline_draft_stage_questions(
                db=session,
                stage_draft_id=uuid.UUID(stage_draft_id),
                questions=questions,
            )
            return f"Successfully updated {len(questions)} questions for stage {stage_draft_id}"

    except Exception as e:
        logger.error(f"Error in update_draft_pipeline_stage_questions: {e}")
        return f"Error updating questions: {str(e)}"


@mcp.tool(
    description='Add a new stage to the pipeline draft. Use format: draft_id|process_type|process_order|is_required|config_json. Example: draft_id|AI_INTERVIEW|4|true|{"duration_minutes":30}'
)
@ensure_json_response
async def add_draft_pipeline_stage(input_data: str) -> str:
    """Add a new stage to pipeline draft"""
    logger.info(f"Tool 'add_draft_pipeline_stage' called with: {input_data}")

    try:
        parts = input_data.split("|")
        if len(parts) < 4:
            return "Error: Invalid format. Use: draft_id|process_type|process_order|is_required|config_json"

        draft_id = parts[0]
        process_type_str = parts[1].strip().upper()
        process_order = int(parts[2])
        is_required = parts[3].strip().lower() == "true"
        config = {}
        if len(parts) > 4:
            config = json.loads(parts[4])

        from app.db.schemas import ProcessType

        process_type = ProcessType[process_type_str]

        async with AsyncSessionLocal() as session:
            await PipelineService.add_stage_to_pipeline_draft(
                db=session,
                draft_id=uuid.UUID(draft_id),
                process_type=process_type,
                process_order=process_order,
                is_required=is_required,
                config=config,
            )
            return f"Successfully added {process_type_str} stage at position {process_order}"

    except Exception as e:
        logger.error(f"Error in add_draft_pipeline_stage: {e}")
        return f"Error adding stage: {str(e)}"


@mcp.tool(
    description="Remove a stage from the pipeline draft. Use format: draft_id|process_type. Example: draft_id|AI_INTERVIEW"
)
@ensure_json_response
async def remove_draft_pipeline_stage(input_data: str) -> str:
    """Remove a stage from pipeline draft"""
    logger.info(f"Tool 'remove_draft_pipeline_stage' called with: {input_data}")

    try:
        parts = input_data.split("|")
        if len(parts) < 2:
            return "Error: Invalid format. Use: draft_id|process_type"

        draft_id = parts[0]
        process_type_str = parts[1].strip().upper()

        from app.db.schemas import ProcessType

        process_type = ProcessType[process_type_str]

        async with AsyncSessionLocal() as session:
            await PipelineService.remove_stage_from_pipeline_draft(
                db=session, draft_id=uuid.UUID(draft_id), process_type=process_type
            )
            return f"Successfully removed {process_type_str} stage from pipeline draft"

    except Exception as e:
        logger.error(f"Error in remove_draft_pipeline_stage: {e}")
        return f"Error removing stage: {str(e)}"


@mcp.tool(
    description="Reorder stages in the pipeline draft. Use format: draft_id|process_type:order,process_type:order. Example: draft_id|PRESCREENING:3,AI_INTERVIEW:4,TECHNICAL_ASSESSMENT:5"
)
@ensure_json_response
async def reorder_draft_pipeline_stages(input_data: str) -> str:
    """Reorder stages in pipeline draft"""
    logger.info(f"Tool 'reorder_draft_pipeline_stages' called with: {input_data}")

    try:
        parts = input_data.split("|")
        if len(parts) < 2:
            return "Error: Invalid format. Use: draft_id|process_type:order,process_type:order"

        draft_id = parts[0]
        stage_orders = {}

        from app.db.schemas import ProcessType

        for stage_spec in parts[1].split(","):
            stage_parts = stage_spec.split(":")
            if len(stage_parts) == 2:
                process_type = ProcessType[stage_parts[0].strip().upper()]
                order = int(stage_parts[1].strip())
                stage_orders[process_type] = order

        async with AsyncSessionLocal() as session:
            await PipelineService.reorder_pipeline_draft_stages(
                db=session, draft_id=uuid.UUID(draft_id), stage_orders=stage_orders
            )
            return (
                f"Successfully reordered {len(stage_orders)} stages in pipeline draft"
            )

    except Exception as e:
        logger.error(f"Error in reorder_draft_pipeline_stages: {e}")
        return f"Error reordering stages: {str(e)}"


@mcp.tool(
    description='Update configuration for a stage in the pipeline draft. Use format: draft_id|process_type|config_json. Example: draft_id|PRESCREENING|{"time_limit_hours":72,"passing_score":80}'
)
@ensure_json_response
async def update_draft_stage_configuration(input_data: str) -> str:
    """Update configuration for a pipeline draft stage"""
    logger.info(f"Tool 'update_draft_stage_configuration' called with: {input_data}")

    try:
        parts = input_data.split("|", 2)  # Split only on first 2 pipes to preserve JSON
        if len(parts) < 3:
            return "Error: Invalid format. Use: draft_id|process_type|config_json"

        draft_id = parts[0]
        process_type_str = parts[1].strip().upper()

        config = json.loads(parts[2])

        from app.db.schemas import ProcessType

        process_type = ProcessType[process_type_str]

        async with AsyncSessionLocal() as session:
            await PipelineService.update_pipeline_draft_stage_config(
                db=session,
                draft_id=uuid.UUID(draft_id),
                process_type=process_type,
                config=config,
            )
            return f"Successfully updated configuration for {process_type_str} stage"

    except Exception as e:
        logger.error(f"Error in update_draft_stage_configuration: {e}")
        return f"Error updating stage configuration: {str(e)}"


@mcp.tool(
    description="Toggle a stage active/inactive in the pipeline draft. Use format: draft_id|process_type|is_active. Example: draft_id|AI_INTERVIEW|false"
)
@ensure_json_response
async def toggle_draft_stage_active(input_data: str) -> str:
    """Toggle stage active/inactive in pipeline draft"""
    logger.info(f"Tool 'toggle_draft_stage_active' called with: {input_data}")

    try:
        # Use utility function for parsing
        parts = parse_pipe_separated_input(
            input_data,
            min_parts=3,
            error_message="Invalid format. Use: draft_id|process_type|is_active",
        )

        draft_id = parts[0]
        process_type_str = parts[1].upper()
        is_active = parse_boolean(parts[2])

        from app.db.schemas import ProcessType

        process_type = ProcessType[process_type_str]

        async with AsyncSessionLocal() as session:
            result = await PipelineService.toggle_pipeline_draft_stage_active(
                db=session,
                draft_id=uuid.UUID(draft_id),
                process_type=process_type,
                is_active=is_active,
            )
            status = "activated" if is_active else "deactivated"

            # Include reordering info in response
            active_stages_info = ", ".join(
                [
                    f"{s['process_type']}:{s['process_order']}"
                    for s in result.get("active_stages_order", [])
                ]
            )

            return (
                f"Successfully {status} {process_type_str} stage in pipeline draft. "
                f"Active stages: {result.get('active_stages_count', 0)}. "
                f"Order: {active_stages_info}"
            )

    except Exception as e:
        logger.error(f"Error in toggle_draft_stage_active: {e}")
        return f"Error toggling stage: {str(e)}"


@mcp.tool(
    description="Configure AI interview stage in draft pipeline. Use format: draft_id|duration_minutes|cutoff_score|link_expiry_days|standard_questions(true/false)|num_contextual_questions. Example: draft_id|20|60|5|false|10 (last param is optional, defaults to 10)"
)
@ensure_json_response
async def configure_draft_ai_interview(input_data: str) -> str:
    """Configure AI interview stage settings in draft pipeline

    Format: draft_id|duration_minutes|cutoff_score|link_expiry_days|standard_questions|num_contextual_questions

    Parameters:
    - draft_id: UUID of the job draft
    - duration_minutes: Interview duration in minutes
    - cutoff_score: Minimum passing score (0-100)
    - link_expiry_days: Number of days before interview link expires
    - standard_questions: true/false - whether to use predefined questions
    - num_contextual_questions: Number of AI-generated contextual questions (optional, defaults to 10)
    """
    logger.info(f"Tool 'configure_draft_ai_interview' called with: {input_data}")

    try:
        parts = input_data.split("|")
        if len(parts) < 5:
            return "Error: Invalid format. Use: draft_id|duration_minutes|cutoff_score|link_expiry_days|standard_questions|num_contextual_questions (last param optional)"

        draft_id = parts[0].strip()
        duration = int(parts[1].strip())
        cutoff = int(parts[2].strip())
        expiry = int(parts[3].strip())
        use_standard = parts[4].strip().lower() == "true"

        # Get number of contextual questions (defaults to 10 if not provided)
        num_contextual = 10
        if len(parts) >= 6:
            num_contextual = int(parts[5].strip())

        config = {
            "number_of_contextual_questions": num_contextual,
            "interview_duration": duration,
            "link_expiry_days": expiry,
            "minimum_cutoff_score": cutoff,
            "standard_questions": use_standard,
        }

        async with AsyncSessionLocal() as session:
            await PipelineService.configure_ai_interview_stage_draft(
                db=session,
                draft_id=uuid.UUID(draft_id),
                config=config,
                questions=None,  # Questions set separately via update_draft_pipeline_stage_questions
            )

            return (
                f"Successfully configured AI interview: "
                f"{duration}min duration, {cutoff}% cutoff, {expiry} days link expiry, "
                f"{num_contextual} contextual questions, standard questions: {use_standard}"
            )

    except Exception as e:
        logger.error(f"Error in configure_draft_ai_interview: {e}")
        return f"Error configuring AI interview: {str(e)}"


@mcp.tool(
    description="Get available Xobin assessments for technical assessment stage in draft pipeline. Use format: draft_id|page_size|page_number (page params optional, defaults to 20 per page). Shows paginated list of assessments."
)
@ensure_json_response
async def get_draft_technical_assessments(input_data: str) -> str:
    """Get available Xobin assessments for a draft's technical assessment stage with pagination

    Format: draft_id|page_size|page_number (optional params)

    Returns a formatted list of available assessments with their IDs and names.
    """
    logger.info(f"Tool 'get_draft_technical_assessments' called with: {input_data}")

    try:
        # Parse input with optional pagination params
        parts = input_data.split("|")
        draft_id = parts[0].strip()
        page_size = int(parts[1].strip()) if len(parts) > 1 else 20
        page_number = int(parts[2].strip()) if len(parts) > 2 else 1

        # Validate pagination params
        if page_size < 1 or page_size > 100:
            return "Error: Page size must be between 1 and 100."
        if page_number < 1:
            return "Error: Page number must be at least 1."

        async with AsyncSessionLocal() as session:
            # Get the pipeline draft
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload
            from app.db.models import PipelineDraft

            pipeline_query = (
                select(PipelineDraft)
                .where(PipelineDraft.job_draft_id == uuid.UUID(draft_id))
                .options(selectinload(PipelineDraft.stages))
            )
            pipeline_draft = (
                await session.execute(pipeline_query)
            ).scalar_one_or_none()

            if not pipeline_draft:
                return "Error: Pipeline draft not found for this job draft."

            # Find TECHNICAL_ASSESSMENT stage
            tech_stage = next(
                (
                    s
                    for s in pipeline_draft.stages
                    if s.process_type == ProcessType.TECHNICAL_ASSESSMENT
                ),
                None,
            )

            if not tech_stage:
                return "Error: Technical Assessment stage not found in pipeline draft."

            if not tech_stage.is_active:
                return "Error: Technical Assessment stage is not active. Please activate it first."

            # Get available assessments from config
            config = tech_stage.process_config or {}
            available_assessments = config.get("available_assessments", [])

            if not available_assessments:
                return "No assessments available. Assessments may still be loading from Xobin. Please try again in a moment."

            # Apply pagination
            total_assessments = len(available_assessments)
            start_idx = (page_number - 1) * page_size
            end_idx = start_idx + page_size
            paginated_assessments = available_assessments[start_idx:end_idx]

            if not paginated_assessments and page_number > 1:
                return f"Error: Page {page_number} is out of range. Total assessments: {total_assessments}."

            # Format the assessments list for display
            assessment_list = []
            for idx, assessment in enumerate(
                paginated_assessments, start=start_idx + 1
            ):
                assessment_id = assessment.get("assessment_id", "N/A")
                assessment_name = assessment.get(
                    "assessment_name", "Unnamed Assessment"
                )
                created_by = assessment.get("created_by", "Unknown")
                assessment_list.append(
                    f"{idx}. ID: {assessment_id} - {assessment_name} (created by: {created_by})"
                )

            total_pages = (total_assessments + page_size - 1) // page_size
            result = f"Available Xobin Assessments (Page {page_number}/{total_pages}, showing {len(paginated_assessments)} of {total_assessments} total):\n\n"
            result += "\n".join(assessment_list)
            result += f"\n\nTo view more: Use draft_id|{page_size}|{page_number + 1}"
            result += "\n\nTo configure an assessment, provide: assessment ID, expiry days (1-10), and proctoring preferences (AI, eye gaze, tab switching, screen recording)."

            return result

    except ValueError as e:
        logger.error(f"Invalid input in get_draft_technical_assessments: {e}")
        return (
            "Error: Invalid pagination parameters. Use: draft_id|page_size|page_number"
        )
    except Exception as e:
        logger.error(f"Error in get_draft_technical_assessments: {e}")
        return f"Error fetching assessments: {str(e)}"


@mcp.tool(
    description="Configure Technical Assessment (Xobin) stage in draft pipeline. Use format: draft_id|assessment_id|assessment_name|expiry_days|ai|eyegaze|offtab|screen_record. Example: draft_id|12345|Python Developer Test|7|true|true|true|false"
)
@ensure_json_response
async def configure_draft_technical_assessment(input_data: str) -> str:
    """Configure Xobin technical assessment stage settings in draft pipeline

    Format: draft_id|assessment_id|assessment_name|expiry_days|ai|eyegaze|offtab|screen_record

    Parameters:
    - draft_id: UUID of the job draft
    - assessment_id: Xobin assessment ID to use
    - assessment_name: Name of the selected assessment
    - expiry_days: Assessment link expiry in days (1-10)
    - ai: Enable AI-based proctoring (true/false)
    - eyegaze: Enable eye gaze tracking (true/false)
    - offtab: Enable tab switching detection (true/false)
    - screen_record: Enable screen recording (true/false)
    """
    logger.info(
        f"Tool 'configure_draft_technical_assessment' called with: {input_data}"
    )

    try:
        parts = input_data.split("|")
        if len(parts) != 8:
            return "Error: Invalid format. Use: draft_id|assessment_id|assessment_name|expiry_days|ai|eyegaze|offtab|screen_record"

        draft_id = parts[0].strip()
        assessment_id = int(parts[1].strip())
        assessment_name = parts[2].strip()
        expiry_days = int(parts[3].strip())

        # Parse proctoring settings
        ai_proctoring = parts[4].strip().lower() == "true"
        eyegaze = parts[5].strip().lower() == "true"
        offtab = parts[6].strip().lower() == "true"
        screen_record = parts[7].strip().lower() == "true"

        proctoring_settings = {
            "ai": ai_proctoring,
            "eyegaze": eyegaze,
            "offtab": offtab,
            "screen_record": screen_record,
        }

        # Build config dict matching the expected structure
        config = {
            "assessment_provider": "xobin",
            "chosen_assessment": {
                "assessment_id": assessment_id,
                "assessment_name": assessment_name,
            },
            "expiry_in_days": expiry_days,
            "proctoring_settings": proctoring_settings,
        }

        async with AsyncSessionLocal() as session:
            await PipelineService.configure_technical_assessment_stage_draft(
                db=session, draft_id=uuid.UUID(draft_id), config=config
            )

            proctoring_enabled = [k for k, v in proctoring_settings.items() if v]
            proctoring_str = (
                ", ".join(proctoring_enabled) if proctoring_enabled else "none"
            )

            return (
                f"Successfully configured Technical Assessment (Xobin): "
                f"Assessment '{assessment_name}' (ID: {assessment_id}), "
                f"{expiry_days} days link expiry, "
                f"Proctoring enabled: {proctoring_str}"
            )

    except ValueError as e:
        logger.error(f"Validation error in configure_draft_technical_assessment: {e}")
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Error in configure_draft_technical_assessment: {e}")
        return f"Error configuring technical assessment: {str(e)}"


@mcp.tool(
    description="Fetches the defined hiring pipeline process steps for a specific job ID."
)
async def get_job_pipeline(input_data: GetJobPipelineInput) -> GetJobPipelineResponse:
    """Get job pipeline steps"""
    logger.info(f"Tool 'get_job_pipeline' called for job_id: {input_data.job_id}")
    async with AsyncSessionLocal() as session:
        return await JobService.get_job_pipeline(db=session, job_id=input_data.job_id)


@mcp.tool(
    description=(
        "Get candidates for a job by pipeline stage. "
        "Format: job_id|stage (client_id is auto-derived from job)\n\n"
        "Available stages:\n"
        "- SOURCED: Pre-sourced candidates from internal database\n"
        "- RANKED: AI-ranked candidates with scores and reasoning\n"
        "- PRESCREENING: Candidates in prescreening (sent/in-progress/passed/failed)\n"
        "- AI_INTERVIEW: AI interview results with transcripts and scores\n"
        "- AI_VIDEO_INTERVIEW: Video interview results from Xobin\n"
        "- TECHNICAL_ASSESSMENT: Technical assessment results from Xobin\n"
        "- CANDIDATES_RECEIVED: Third-last stage candidates\n"
        "- CLIENT_INTERVIEW: Candidates in client interview stage\n"
        "- SELECTED_HIRED: Selected/hired candidates\n"
        "- JOINED: Candidates who joined\n\n"
        "If stage is omitted, returns pipeline statistics for all stages.\n"
        "Note: Some stages return paginated results (first page by default)."
    )
)
@ensure_json_response
async def get_job_candidates_by_stage(input_data: str) -> str:
    """Unified tool to get candidates by stage using exact API service methods"""
    logger.info(f"Tool 'get_job_candidates_by_stage' called with: {input_data}")

    try:
        # Parse: job_id or job_id|STAGE or job_id|STAGE|client_id
        parts = input_data.split("|")
        if len(parts) < 1:
            return json.dumps({"error": "Invalid format. Use: job_id or job_id|STAGE or job_id|STAGE|client_id"}, indent=2)

        job_id = parts[0].strip()
        stage = parts[1].strip().upper() if len(parts) > 1 else None
        client_id = parts[2].strip() if len(parts) > 2 else None

        async with AsyncSessionLocal() as session:
            # If no stage specified, return pipeline statistics
            if not stage:
                stats = await PipelineService.get_pipeline_statistics(
                    db=session, job_posting_id=uuid.UUID(job_id)
                )
                return json.dumps(stats, indent=2, default=str)

            # Route to appropriate service based on stage
            if stage in ["SOURCED", "SOURCING"]:
                from app.candidates.services.sourcing_service import CandidateSourcingService
                result = await CandidateSourcingService.get_sourced_candidates_detailed(
                    db=session, job_id=uuid.UUID(job_id)
                )
                response_data = result.dict() if hasattr(result, 'dict') else result
                response_data["stage"] = "SOURCED"
                return json.dumps(response_data, indent=2, default=str)

            elif stage in ["RANKED", "RANKING"]:
                from app.candidates.services.ranking_service import RankedCandidateService
                service = RankedCandidateService(session)
                result = await service.get_ranked_candidates(uuid.UUID(job_id))
                response_data = result.dict() if hasattr(result, 'dict') else result
                response_data["stage"] = "RANKED"
                return json.dumps(response_data, indent=2, default=str)

            elif stage in ["PRESCREENING", "PRE_SCREENING"]:
                from app.candidates.services.ranking_service import RankedCandidateService
                service = RankedCandidateService(session)
                result = await service.get_prescreening_candidates(uuid.UUID(job_id))
                response_data = result.dict() if hasattr(result, 'dict') else result
                response_data["stage"] = "PRESCREENING"
                return json.dumps(response_data, indent=2, default=str)

            elif stage == "AI_INTERVIEW":
                from app.ai_interview.services.ai_interview_results_service import get_paginated_interview_results                
                result = await get_paginated_interview_results(
                    db=session,
                    page_number=1,
                    per_page=10,
                    job_posting_id=uuid.UUID(job_id)
                )
                response_data = result.dict() if hasattr(result, 'dict') else result
                response_data["stage"] = "AI_INTERVIEW"
                return json.dumps(response_data, indent=2, default=str)

            elif stage == "AI_VIDEO_INTERVIEW":
                from app.technical_interview_xobin.services.video_interview_result_service import fetch_ai_video_interviews_for_job
                result = await fetch_ai_video_interviews_for_job(
                    db=session,
                    job_posting_id=uuid.UUID(job_id),
                    page_number=1,
                    page_size=10
                )
                response_data = result.dict() if hasattr(result, 'dict') else result
                response_data["stage"] = "AI_VIDEO_INTERVIEW"
                return json.dumps(response_data, indent=2, default=str)

            elif stage == "TECHNICAL_ASSESSMENT":
                from app.technical_interview_xobin.services.technical_interview_result_service import fetch_technical_interviews_for_job
                result = await fetch_technical_interviews_for_job(
                    db=session,
                    job_posting_id=uuid.UUID(job_id),
                    page_number=1,
                    page_size=10
                )
                response_data = result.dict() if hasattr(result, 'dict') else result
                response_data["stage"] = "TECHNICAL_ASSESSMENT"
                return json.dumps(response_data, indent=2, default=str)

            elif stage == "CANDIDATES_RECEIVED":
                # Auto-derive client_id from job if not provided
                if not client_id:
                    from app.db.models import JobPosting
                    from sqlalchemy.orm import selectinload
                    stmt = select(JobPosting).options(selectinload(JobPosting.job_requirement)).where(JobPosting.job_posting_id == uuid.UUID(job_id))
                    result = await session.execute(stmt)
                    job = result.scalars().first()
                    if job and job.job_requirement:
                        client_id = str(job.job_requirement.client_id)
                    else:
                        return json.dumps({"error": "Could not derive client_id from job. Use format: job_id|CANDIDATES_RECEIVED|client_id"}, indent=2)

                from app.client_dashboard.services.client_dashboard_service import ClientDashboardService
                # Create a mock user object for MCP tool context
                mock_user = type('MockUser', (), {'client_id': uuid.UUID(client_id), 'user_id': None, 'role': UserRole.RECRUITER})()
                service = ClientDashboardService(session, mock_user, client_id_override=uuid.UUID(client_id))
                result = await service.get_third_last_stage_applications(
                    job_posting_id=uuid.UUID(job_id),
                    page_number=1,
                    page_size=10
                )
                response_data = result.dict() if hasattr(result, 'dict') else result
                response_data["stage"] = "CANDIDATES_RECEIVED"
                return json.dumps(response_data, indent=2, default=str)

            elif stage == "CLIENT_INTERVIEW":
                # Auto-derive client_id from job if not provided
                if not client_id:
                    from app.db.models import JobPosting
                    from sqlalchemy.orm import selectinload
                    stmt = select(JobPosting).options(selectinload(JobPosting.job_requirement)).where(JobPosting.job_posting_id == uuid.UUID(job_id))
                    result = await session.execute(stmt)
                    job = result.scalars().first()
                    if job and job.job_requirement:
                        client_id = str(job.job_requirement.client_id)
                    else:
                        return json.dumps({"error": "Could not derive client_id from job. Use format: job_id|CLIENT_INTERVIEW|client_id"}, indent=2)

                from app.client_dashboard.services.client_dashboard_service import ClientDashboardService
                mock_user = type('MockUser', (), {'client_id': uuid.UUID(client_id), 'user_id': None, 'role': UserRole.RECRUITER})()
                service = ClientDashboardService(session, mock_user, client_id_override=uuid.UUID(client_id))
                result = await service.get_client_interview_candidates(
                    job_posting_id=uuid.UUID(job_id),
                    page=1,
                    page_size=10
                )
                response_data = result.dict() if hasattr(result, 'dict') else result
                response_data["stage"] = "CLIENT_INTERVIEW"
                return json.dumps(response_data, indent=2, default=str)

            elif stage == "SELECTED_HIRED":
                # Auto-derive client_id from job if not provided
                if not client_id:
                    from app.db.models import JobPosting
                    from sqlalchemy.orm import selectinload
                    stmt = select(JobPosting).options(selectinload(JobPosting.job_requirement)).where(JobPosting.job_posting_id == uuid.UUID(job_id))
                    result = await session.execute(stmt)
                    job = result.scalars().first()
                    if job and job.job_requirement:
                        client_id = str(job.job_requirement.client_id)
                    else:
                        return json.dumps({"error": "Could not derive client_id from job. Use format: job_id|SELECTED_HIRED|client_id"}, indent=2)

                from app.client_dashboard.services.client_dashboard_service import ClientDashboardService
                mock_user = type('MockUser', (), {'client_id': uuid.UUID(client_id), 'user_id': None, 'role': UserRole.RECRUITER})()
                service = ClientDashboardService(session, mock_user, client_id_override=uuid.UUID(client_id))
                result = await service.get_selected_hired_candidates(
                    job_posting_id=uuid.UUID(job_id),
                    page=1,
                    page_size=10
                )
                response_data = result.dict() if hasattr(result, 'dict') else result
                response_data["stage"] = "SELECTED_HIRED"
                return json.dumps(response_data, indent=2, default=str)

            elif stage == "JOINED":
                # Auto-derive client_id from job if not provided
                if not client_id:
                    from app.db.models import JobPosting
                    from sqlalchemy.orm import selectinload
                    stmt = select(JobPosting).options(selectinload(JobPosting.job_requirement)).where(JobPosting.job_posting_id == uuid.UUID(job_id))
                    result = await session.execute(stmt)
                    job = result.scalars().first()
                    if job and job.job_requirement:
                        client_id = str(job.job_requirement.client_id)
                    else:
                        return json.dumps({"error": "Could not derive client_id from job. Use format: job_id|JOINED|client_id"}, indent=2)

                from app.client_dashboard.services.client_dashboard_service import ClientDashboardService
                mock_user = type('MockUser', (), {'client_id': uuid.UUID(client_id), 'user_id': None, 'role': UserRole.RECRUITER})()
                service = ClientDashboardService(session, mock_user, client_id_override=uuid.UUID(client_id))
                result = await service.get_joined_candidates(
                    job_posting_id=uuid.UUID(job_id),
                    page=1,
                    page_size=10
                )
                response_data = result.dict() if hasattr(result, 'dict') else result
                response_data["stage"] = "JOINED"
                return json.dumps(response_data, indent=2, default=str)

            else:
                return json.dumps({
                    "error": f"Unknown stage: {stage}",
                    "available_stages": ["SOURCED", "RANKED", "PRESCREENING", "AI_INTERVIEW",
                                         "AI_VIDEO_INTERVIEW", "TECHNICAL_ASSESSMENT",
                                         "CANDIDATES_RECEIVED", "CLIENT_INTERVIEW",
                                         "SELECTED_HIRED", "JOINED"]
                }, indent=2)

    except Exception as e:
        logger.error(f"Error in get_job_candidates_by_stage: {e}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool(
    description="Sends pre-screening test invitations to a list of sourced candidates for a specific job."
)
@ensure_json_response
async def send_prescreening_invites(
    input_data: SendPrescreeningInvitesInput,
) -> Union[SendPrescreeningInvitesResponse, str]:
    """Send prescreening invitations"""
    logger.info(
        f"Tool 'send_prescreening_invites' called for job_id: {input_data.job_id}"
    )
    async with AsyncSessionLocal() as session:
        return await CandidateSourcingService.send_prescreening_invites(
            db=session, job_id=input_data.job_id, candidate_ids=input_data.candidate_ids
        )


# =========================================================================
#                   CANDIDATE ANALYTICS TOOLS
# =========================================================================


@mcp.tool(
    description=(
        "Display a welcome card with helpful information for the candidate. "
        "Use this when the candidate greets you (hi, hello, hey, what can you do) or starts a new conversation. "
        "This shows a friendly welcome message with example questions they can ask."
    )
)
async def show_candidate_welcome_card() -> str:
    """
    Display a welcome card component with helpful information for candidates.
    Used for greetings and initial interactions.
    """
    logger.info("Tool 'show_candidate_welcome_card' called")
    try:
        return json.dumps({
            "trigger": "show_candidate_welcome_card",
            "message": "I'm here to help you track and manage your job applications!",
            "example_questions": [
                "What jobs have I applied for?",
                "Do I have any interviews scheduled today?",
                "What is the status of my applications?",
                "Show me all companies I've applied to",
                "Do I have any pending assessments?",
                "When is my next interview?"
            ],
            "capabilities": [
                {
                    "title": "Application Status",
                    "description": "Track all your job applications and their current status",
                    "icon": "briefcase"
                },
                {
                    "title": "Interviews & Assessments",
                    "description": "View upcoming interviews and assessment schedules",
                    "icon": "calendar"
                },
                {
                    "title": "Profile & Documents",
                    "description": "Check your uploaded documents and profile information",
                    "icon": "user"
                }
            ]
        })
    except Exception as e:
        logger.error(f"Error in show_candidate_welcome_card: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description=(
        "Retrieves information about a specific candidate's profile, documents, or job applications. "
        "This tool accepts a natural language question and the authenticated candidate's ID, "
        "generates a secure read-only SQL query, executes it against the database, "
        "and returns the results.\n\n"
        "Scope: Only returns data for the specified candidate_id. Cannot access other candidates' information.\n"
        f"Tables available: {TASC_SCHEMA}.candidates, {TASC_SCHEMA}.documents, {TASC_SCHEMA}.applications, {TASC_SCHEMA}.job_posting\n\n"
        "Example questions:\n"
        "- 'What jobs have I applied to?'\n"
        "- 'Show me my uploaded documents'\n"
        "- 'What is my profile information?'\n"
        "- 'What is the status of my applications?'"
    )
)
async def answer_candidate_question(
    input_data: RunCandidateQueryInput,
) -> CandidateSqlQueryResponse:
    """
    Tool for retrieving candidate-specific information through SQL queries.
    All queries are automatically scoped to the provided candidate_id for security.
    """
    logger.info(
        f"Tool 'answer_candidate_question' invoked for candidate: {input_data.candidate_id}"
    )

    async with AsyncSessionLocal() as session:
        return await CandidateQueryService.run_candidate_query(
            db=session,
            user_prompt=input_data.user_prompt,
            secure_candidate_id=input_data.candidate_id,
            db_schema=SchemaIndex.CANDIDATE_DB_SCHEMA_FOR_TOOL,
        )


# =======================================================================
#                    NEW CANDIDATE-SPECIFIC TOOLS
# =======================================================================


@mcp.tool(
    description=(
        "Get all interview results for a candidate including AI interviews, video interviews, "
        "and client interviews with scores, feedback, and status. "
        "Returns structured data about all interview types the candidate has participated in."
    )
)
async def get_candidate_interview_results(candidate_id: str) -> str:
    """
    Retrieve all interview results for a candidate including:
    - AI Interview: scores, strengths, weaknesses, status
    - Video Interview: Xobin scores, recommendation
    - Client Interview: booking status, feedback, result
    """
    logger.info(f"Tool 'get_candidate_interview_results' called for candidate: {candidate_id}")

    try:
        candidate_uuid = uuid.UUID(candidate_id)

        async with AsyncSessionLocal() as session:
            # Query for AI/Video interviews from response_sessions
            ai_video_query = f"""
                SELECT
                    jp.title as job_title,
                    jpp.process_type,
                    rs.status as session_status,
                    rs.response_details,
                    rs.created_at as interview_date
                FROM {TASC_SCHEMA}.response_sessions rs
                JOIN {TASC_SCHEMA}.applications a ON rs.application_id = a.application_id
                JOIN {TASC_SCHEMA}.job_posting_process jpp ON rs.process_id = jpp.process_id
                JOIN {TASC_SCHEMA}.job_posting jp ON a.job_posting_id = jp.job_posting_id
                WHERE a.candidate_id = :candidate_id
                  AND jpp.process_type IN ('AI_INTERVIEW', 'AI_VIDEO_INTERVIEW', 'PRESCREENING')
                ORDER BY rs.created_at DESC
            """

            from sqlalchemy import text
            ai_video_result = await session.execute(
                text(ai_video_query),
                {"candidate_id": candidate_uuid}
            )
            ai_video_rows = ai_video_result.fetchall()

            # Query for client interviews from interview_bookings
            client_interview_query = f"""
                SELECT
                    jp.title as job_title,
                    ib.status as booking_status,
                    ib.result,
                    ib.feedback,
                    ib.google_meet_link,
                    its.slot_start_time as interview_time,
                    its.slot_end_time,
                    ib.completed_at
                FROM {TASC_SCHEMA}.interview_bookings ib
                JOIN {TASC_SCHEMA}.interview_time_slots its ON ib.slot_id = its.slot_id
                LEFT JOIN {TASC_SCHEMA}.applications a ON ib.application_id = a.application_id
                LEFT JOIN {TASC_SCHEMA}.job_posting jp ON a.job_posting_id = jp.job_posting_id
                WHERE ib.candidate_id = :candidate_id
                ORDER BY its.slot_start_time DESC
            """

            client_result = await session.execute(
                text(client_interview_query),
                {"candidate_id": candidate_uuid}
            )
            client_rows = client_result.fetchall()

            # Format AI/Video interview results
            ai_interviews = []
            video_interviews = []
            prescreening_results = []

            for row in ai_video_rows:
                details = row.response_details or {}
                interview_data = {
                    "job_title": row.job_title,
                    "status": row.session_status,
                    "date": row.interview_date.isoformat() if row.interview_date else None,
                }

                if row.process_type == "AI_INTERVIEW":
                    interview_data.update({
                        "overall_score": details.get("overall_score"),
                        "strengths": details.get("strengths"),
                        "weaknesses": details.get("weaknesses"),
                        "recommendation": details.get("recommendation"),
                    })
                    ai_interviews.append(interview_data)

                elif row.process_type == "AI_VIDEO_INTERVIEW":
                    xobin_report = details.get("xobin_report_response", {})
                    interview_data.update({
                        "overall_percentage": xobin_report.get("overall_percentage"),
                        "integrity_score": xobin_report.get("integrity_score"),
                        "recommendation": xobin_report.get("recommendation"),
                    })
                    video_interviews.append(interview_data)

                elif row.process_type == "PRESCREENING":
                    interview_data.update({
                        "prescreening_score": details.get("score"),
                        "responses": details.get("responses"),
                    })
                    prescreening_results.append(interview_data)

            # Format client interview results
            client_interviews = []
            for row in client_rows:
                client_interviews.append({
                    "job_title": row.job_title,
                    "booking_status": row.booking_status,
                    "result": row.result,
                    "feedback": row.feedback,
                    "interview_time": row.interview_time.isoformat() if row.interview_time else None,
                    "meeting_link": row.google_meet_link,
                    "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                })

            result = {
                "ai_interviews": ai_interviews,
                "video_interviews": video_interviews,
                "prescreening_results": prescreening_results,
                "client_interviews": client_interviews,
                "summary": {
                    "total_ai_interviews": len(ai_interviews),
                    "total_video_interviews": len(video_interviews),
                    "total_client_interviews": len(client_interviews),
                    "total_prescreenings": len(prescreening_results),
                }
            }

            return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Error in get_candidate_interview_results: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description=(
        "Get technical assessment results for a candidate including Xobin scores, "
        "skill breakdown, and integrity scores. Returns detailed assessment data "
        "for all technical tests the candidate has taken."
    )
)
async def get_candidate_assessment_results(candidate_id: str) -> str:
    """
    Retrieve technical assessment results including:
    - Assessment name, overall percentage, cut-off score
    - Skill-wise scores breakdown
    - Integrity score
    - Status (passed/failed)
    """
    logger.info(f"Tool 'get_candidate_assessment_results' called for candidate: {candidate_id}")

    try:
        candidate_uuid = uuid.UUID(candidate_id)

        async with AsyncSessionLocal() as session:
            query = f"""
                SELECT
                    jp.title as job_title,
                    jpp.process_type,
                    rs.status as session_status,
                    rs.response_details,
                    rs.created_at as assessment_date,
                    rs.updated_at
                FROM {TASC_SCHEMA}.response_sessions rs
                JOIN {TASC_SCHEMA}.applications a ON rs.application_id = a.application_id
                JOIN {TASC_SCHEMA}.job_posting_process jpp ON rs.process_id = jpp.process_id
                JOIN {TASC_SCHEMA}.job_posting jp ON a.job_posting_id = jp.job_posting_id
                WHERE a.candidate_id = :candidate_id
                  AND jpp.process_type = 'TECHNICAL_ASSESSMENT'
                ORDER BY rs.created_at DESC
            """

            from sqlalchemy import text
            result = await session.execute(text(query), {"candidate_id": candidate_uuid})
            rows = result.fetchall()

            assessments = []
            for row in rows:
                details = row.response_details or {}
                xobin_report = details.get("xobin_report_response", {})

                assessment_data = {
                    "job_title": row.job_title,
                    "status": row.session_status,
                    "assessment_date": row.assessment_date.isoformat() if row.assessment_date else None,
                    "overall_percentage": xobin_report.get("overall_percentage"),
                    "cut_off_score": xobin_report.get("cut_off_score"),
                    "integrity_score": xobin_report.get("integrity_score"),
                    "skill_scores": xobin_report.get("skill_wise_scores", []),
                    "time_taken": xobin_report.get("time_taken"),
                    "passed": row.session_status == "COMPLETED" and
                              (xobin_report.get("overall_percentage", 0) or 0) >= (xobin_report.get("cut_off_score", 0) or 0),
                }
                assessments.append(assessment_data)

            return json.dumps({
                "assessments": assessments,
                "total_assessments": len(assessments),
                "passed_count": sum(1 for a in assessments if a.get("passed")),
                "failed_count": sum(1 for a in assessments if not a.get("passed") and a.get("status") == "COMPLETED"),
            }, default=str)

    except Exception as e:
        logger.error(f"Error in get_candidate_assessment_results: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description=(
        "Get application timeline showing how long a candidate spent in each pipeline stage. "
        "Returns stage-by-stage timeline with entry/exit dates and time spent in each stage."
    )
)
async def get_candidate_application_timeline(candidate_id: str, job_posting_id: Optional[str] = None) -> str:
    """
    Retrieve application timeline including:
    - Stage-by-stage timeline with entry/exit dates
    - Time spent in each stage
    - Current stage and how long they've been there
    - Uses Application.pipeline_history JSONB field
    """
    logger.info(f"Tool 'get_candidate_application_timeline' called for candidate: {candidate_id}")

    try:
        candidate_uuid = uuid.UUID(candidate_id)

        async with AsyncSessionLocal() as session:
            query = f"""
                SELECT
                    a.application_id,
                    jp.title as job_title,
                    c.company_name,
                    a.status as current_status,
                    a.applied_at,
                    a.updated_at,
                    a.pipeline_history
                FROM {TASC_SCHEMA}.applications a
                JOIN {TASC_SCHEMA}.job_posting jp ON a.job_posting_id = jp.job_posting_id
                LEFT JOIN {TASC_SCHEMA}.job_requirement jr ON jp.job_requirement_id = jr.job_requirement_id
                LEFT JOIN {TASC_SCHEMA}.clients cl ON jr.client_id = cl.client_id
                LEFT JOIN {TASC_SCHEMA}.companies c ON cl.company_id = c.company_id
                WHERE a.candidate_id = :candidate_id
            """

            params = {"candidate_id": candidate_uuid}

            if job_posting_id:
                query += " AND a.job_posting_id = :job_posting_id"
                params["job_posting_id"] = uuid.UUID(job_posting_id)

            query += " ORDER BY a.applied_at DESC"

            from sqlalchemy import text
            result = await session.execute(text(query), params)
            rows = result.fetchall()

            timelines = []
            for row in rows:
                pipeline_history = row.pipeline_history or []

                # Calculate time in each stage
                stage_durations = []
                for i, entry in enumerate(pipeline_history):
                    stage_data = {
                        "stage": entry.get("stage") or entry.get("process_type"),
                        "status": entry.get("status"),
                        "entered_at": entry.get("timestamp") or entry.get("entered_at"),
                        "result": entry.get("result"),
                    }

                    # Calculate duration if there's a next stage
                    if i < len(pipeline_history) - 1:
                        try:
                            entered = datetime.fromisoformat(stage_data["entered_at"].replace("Z", "+00:00"))
                            next_entered = datetime.fromisoformat(
                                (pipeline_history[i + 1].get("timestamp") or
                                 pipeline_history[i + 1].get("entered_at")).replace("Z", "+00:00")
                            )
                            duration = next_entered - entered
                            stage_data["duration_days"] = duration.days
                            stage_data["duration_hours"] = duration.seconds // 3600
                        except (ValueError, TypeError, AttributeError):
                            pass

                    stage_durations.append(stage_data)

                # Calculate time in current stage
                current_stage_duration = None
                if row.updated_at and pipeline_history:
                    try:
                        last_entry_time = pipeline_history[-1].get("timestamp") or pipeline_history[-1].get("entered_at")
                        if last_entry_time:
                            last_time = datetime.fromisoformat(last_entry_time.replace("Z", "+00:00"))
                            now = datetime.now(timezone.utc)
                            duration = now - last_time
                            current_stage_duration = {
                                "days": duration.days,
                                "hours": duration.seconds // 3600,
                            }
                    except (ValueError, TypeError, AttributeError):
                        pass

                timelines.append({
                    "application_id": str(row.application_id),
                    "job_title": row.job_title,
                    "company_name": row.company_name,
                    "current_status": row.current_status,
                    "applied_at": row.applied_at.isoformat() if row.applied_at else None,
                    "last_updated": row.updated_at.isoformat() if row.updated_at else None,
                    "stage_history": stage_durations,
                    "time_in_current_stage": current_stage_duration,
                })

            return json.dumps({
                "timelines": timelines,
                "total_applications": len(timelines),
            }, default=str)

    except Exception as e:
        logger.error(f"Error in get_candidate_application_timeline: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description=(
        "Get recruiter contact information for a candidate's applications. "
        "Returns the recruiter name, email, and contact details for each application."
    )
)
async def get_candidate_recruiter_contact(candidate_id: str) -> str:
    """
    Retrieve recruiter contact information for each application including:
    - Recruiter name and email
    - Company name
    - Job title
    - Application status
    """
    logger.info(f"Tool 'get_candidate_recruiter_contact' called for candidate: {candidate_id}")

    try:
        candidate_uuid = uuid.UUID(candidate_id)

        async with AsyncSessionLocal() as session:
            query = f"""
                SELECT DISTINCT
                    jp.title as job_title,
                    c.company_name,
                    a.status as application_status,
                    u.first_name as recruiter_first_name,
                    u.last_name as recruiter_last_name,
                    u.email as recruiter_email,
                    a.applied_at
                FROM {TASC_SCHEMA}.applications a
                JOIN {TASC_SCHEMA}.job_posting jp ON a.job_posting_id = jp.job_posting_id
                LEFT JOIN {TASC_SCHEMA}.recruiters r ON jp.recruiter_id = r.recruiter_id
                LEFT JOIN {TASC_SCHEMA}.users u ON r.user_id = u.user_id
                LEFT JOIN {TASC_SCHEMA}.job_requirement jr ON jp.job_requirement_id = jr.job_requirement_id
                LEFT JOIN {TASC_SCHEMA}.clients cl ON jr.client_id = cl.client_id
                LEFT JOIN {TASC_SCHEMA}.companies c ON cl.company_id = c.company_id
                WHERE a.candidate_id = :candidate_id
                ORDER BY a.applied_at DESC
            """

            from sqlalchemy import text
            result = await session.execute(text(query), {"candidate_id": candidate_uuid})
            rows = result.fetchall()

            contacts = []
            for row in rows:
                recruiter_name = None
                if row.recruiter_first_name or row.recruiter_last_name:
                    recruiter_name = f"{row.recruiter_first_name or ''} {row.recruiter_last_name or ''}".strip()

                contacts.append({
                    "job_title": row.job_title,
                    "company_name": row.company_name,
                    "application_status": row.application_status,
                    "recruiter_name": recruiter_name,
                    "recruiter_email": row.recruiter_email,
                    "applied_at": row.applied_at.isoformat() if row.applied_at else None,
                })

            return json.dumps({
                "contacts": contacts,
                "total_applications": len(contacts),
            }, default=str)

    except Exception as e:
        logger.error(f"Error in get_candidate_recruiter_contact: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description=(
        "Get offer details for a candidate's applications that reached the offer stage. "
        "Returns offer status, date, job title, and company information."
    )
)
async def get_candidate_offer_details(candidate_id: str) -> str:
    """
    Retrieve offer details including:
    - Offer status (extended, accepted, rejected, revoked)
    - Offer date
    - Job title and company
    - From Application.client_feedback_history
    """
    logger.info(f"Tool 'get_candidate_offer_details' called for candidate: {candidate_id}")

    try:
        candidate_uuid = uuid.UUID(candidate_id)

        # Define offer-related statuses
        offer_statuses = (
            'OFFER_EXTENDED', 'OFFER_ACCEPTED', 'OFFER_REJECTED',
            'OFFER_REVOKED', 'HIRED', 'JOINED'
        )

        async with AsyncSessionLocal() as session:
            query = f"""
                SELECT
                    a.application_id,
                    jp.title as job_title,
                    c.company_name,
                    a.status as current_status,
                    a.client_feedback_history,
                    a.pipeline_history,
                    a.updated_at
                FROM {TASC_SCHEMA}.applications a
                JOIN {TASC_SCHEMA}.job_posting jp ON a.job_posting_id = jp.job_posting_id
                LEFT JOIN {TASC_SCHEMA}.job_requirement jr ON jp.job_requirement_id = jr.job_requirement_id
                LEFT JOIN {TASC_SCHEMA}.clients cl ON jr.client_id = cl.client_id
                LEFT JOIN {TASC_SCHEMA}.companies c ON cl.company_id = c.company_id
                WHERE a.candidate_id = :candidate_id
                  AND a.status IN :offer_statuses
                ORDER BY a.updated_at DESC
            """

            from sqlalchemy import text
            result = await session.execute(
                text(query.replace(":offer_statuses", str(offer_statuses))),
                {"candidate_id": candidate_uuid}
            )
            rows = result.fetchall()

            offers = []
            for row in rows:
                feedback_history = row.client_feedback_history or []
                pipeline_history = row.pipeline_history or []

                # Find offer-related entries in feedback history
                offer_entry = None
                for entry in reversed(feedback_history):
                    if entry.get("type") in ["OFFER", "HIRE", "offer", "hire"]:
                        offer_entry = entry
                        break

                # Find offer timestamp from pipeline history
                offer_timestamp = None
                for entry in pipeline_history:
                    if entry.get("status") in offer_statuses or entry.get("stage") in ["FINAL_EVALUATION"]:
                        offer_timestamp = entry.get("timestamp") or entry.get("entered_at")

                offers.append({
                    "application_id": str(row.application_id),
                    "job_title": row.job_title,
                    "company_name": row.company_name,
                    "offer_status": row.current_status,
                    "offer_date": offer_timestamp,
                    "offer_details": offer_entry,
                    "last_updated": row.updated_at.isoformat() if row.updated_at else None,
                })

            # Also check for any applications with offer-related feedback even if not in offer status
            pending_query = f"""
                SELECT
                    a.application_id,
                    jp.title as job_title,
                    c.company_name,
                    a.status as current_status,
                    a.client_feedback_history,
                    a.updated_at
                FROM {TASC_SCHEMA}.applications a
                JOIN {TASC_SCHEMA}.job_posting jp ON a.job_posting_id = jp.job_posting_id
                LEFT JOIN {TASC_SCHEMA}.job_requirement jr ON jp.job_requirement_id = jr.job_requirement_id
                LEFT JOIN {TASC_SCHEMA}.clients cl ON jr.client_id = cl.client_id
                LEFT JOIN {TASC_SCHEMA}.companies c ON cl.company_id = c.company_id
                WHERE a.candidate_id = :candidate_id
                  AND a.client_feedback_history IS NOT NULL
                  AND jsonb_array_length(a.client_feedback_history) > 0
            """

            return json.dumps({
                "offers": offers,
                "total_offers": len(offers),
                "offer_statuses_explained": {
                    "OFFER_EXTENDED": "An offer has been extended to you",
                    "OFFER_ACCEPTED": "You have accepted the offer",
                    "OFFER_REJECTED": "The offer was declined",
                    "HIRED": "You have been hired",
                    "JOINED": "You have joined the company",
                }
            }, default=str)

    except Exception as e:
        logger.error(f"Error in get_candidate_offer_details: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description=(
        "Get pipeline progress summary showing which stages a candidate has passed, failed, "
        "or is currently in for each application. Useful for answering questions like "
        "'How many jobs have I passed ranking stage?' or 'What's my progress?'"
    )
)
async def get_candidate_pipeline_progress(candidate_id: str) -> str:
    """
    Retrieve pipeline progress summary including:
    - Stages passed (e.g., "Passed: Sourcing, Ranking, Prescreening")
    - Current stage
    - Stages remaining
    - Overall progress percentage
    """
    logger.info(f"Tool 'get_candidate_pipeline_progress' called for candidate: {candidate_id}")

    try:
        candidate_uuid = uuid.UUID(candidate_id)

        # Define pipeline stages in order
        PIPELINE_STAGES = [
            "SOURCING", "RANKING", "PRESCREENING", "AI_INTERVIEW",
            "TECHNICAL_ASSESSMENT", "AI_VIDEO_INTERVIEW", "CLIENT_INTERVIEW", "FINAL_EVALUATION"
        ]

        # Status to stage mapping
        STATUS_TO_STAGE = {
            "SOURCED": 0,
            "RANKED": 1,
            "SCREENING_SENT": 2, "SCREENING_IN_PROGRESS": 2, "SCREENING_PASSED": 2, "SCREENING_FAILED": 2,
            "AI_ITRW_LNK_SNT": 3, "AI_ITRW_IP": 3, "AI_ITRW_PSD": 3, "AI_ITRW_FLD": 3,
            "TECH_ITRW_SCHD": 4, "TECH_ITRW_IP": 4, "TECH_ITRW_PSD": 4, "TECH_ITRW_FLD": 4,
            "VIDEO_ITRW_SCHD": 5, "VIDEO_ITRW_IP": 5, "VIDEO_ITRW_PSD": 5, "VIDEO_ITRW_FLD": 5,
            "SHORTLISTED": 6, "CLIENT_ITRW_PENDING": 6, "CLIENT_ITRW_SELECTED": 6,
            "INTERVIEW_SCHEDULED": 6, "CLIENT_INTERVIEW_SCHEDULED": 6,
            "CLIENT_INTERVIEW_COMPLETED": 6, "CLIENT_INTERVIEW_PASSED": 6, "REJECTED": 6,
            "OFFER_EXTENDED": 7, "OFFER_ACCEPTED": 7, "OFFER_REJECTED": 7, "HIRED": 7, "JOINED": 7,
        }

        # Failed statuses
        FAILED_STATUSES = {
            "SCREENING_FAILED", "AI_ITRW_FLD", "TECH_ITRW_FLD", "VIDEO_ITRW_FLD",
            "REJECTED", "OFFER_REJECTED"
        }

        async with AsyncSessionLocal() as session:
            query = f"""
                SELECT
                    a.application_id,
                    jp.title as job_title,
                    c.company_name,
                    a.status as current_status,
                    a.pipeline_history,
                    a.applied_at
                FROM {TASC_SCHEMA}.applications a
                JOIN {TASC_SCHEMA}.job_posting jp ON a.job_posting_id = jp.job_posting_id
                LEFT JOIN {TASC_SCHEMA}.job_requirement jr ON jp.job_requirement_id = jr.job_requirement_id
                LEFT JOIN {TASC_SCHEMA}.clients cl ON jr.client_id = cl.client_id
                LEFT JOIN {TASC_SCHEMA}.companies c ON cl.company_id = c.company_id
                WHERE a.candidate_id = :candidate_id
                ORDER BY a.applied_at DESC
            """

            from sqlalchemy import text
            result = await session.execute(text(query), {"candidate_id": candidate_uuid})
            rows = result.fetchall()

            progress_data = []
            stage_counts = {stage: {"passed": 0, "in_progress": 0, "failed": 0} for stage in PIPELINE_STAGES}

            for row in rows:
                current_status = row.current_status
                current_stage_index = STATUS_TO_STAGE.get(current_status, 0)
                is_failed = current_status in FAILED_STATUSES

                # Determine passed stages
                passed_stages = PIPELINE_STAGES[:current_stage_index] if not is_failed else PIPELINE_STAGES[:current_stage_index]
                current_stage = PIPELINE_STAGES[current_stage_index] if current_stage_index < len(PIPELINE_STAGES) else "COMPLETED"
                remaining_stages = PIPELINE_STAGES[current_stage_index + 1:] if not is_failed else []

                # Calculate progress percentage
                progress_pct = (current_stage_index / len(PIPELINE_STAGES)) * 100
                if current_status in ["HIRED", "JOINED"]:
                    progress_pct = 100

                # Update stage counts
                for stage in passed_stages:
                    stage_counts[stage]["passed"] += 1
                if not is_failed and current_stage in stage_counts:
                    stage_counts[current_stage]["in_progress"] += 1
                elif is_failed and current_stage in stage_counts:
                    stage_counts[current_stage]["failed"] += 1

                progress_data.append({
                    "application_id": str(row.application_id),
                    "job_title": row.job_title,
                    "company_name": row.company_name,
                    "current_status": current_status,
                    "current_stage": current_stage,
                    "passed_stages": passed_stages,
                    "remaining_stages": remaining_stages,
                    "is_failed": is_failed,
                    "progress_percentage": round(progress_pct, 1),
                    "applied_at": row.applied_at.isoformat() if row.applied_at else None,
                })

            # Summary statistics
            summary = {
                "total_applications": len(progress_data),
                "passed_ranking": sum(1 for p in progress_data if "RANKING" in p["passed_stages"]),
                "passed_prescreening": sum(1 for p in progress_data if "PRESCREENING" in p["passed_stages"]),
                "passed_ai_interview": sum(1 for p in progress_data if "AI_INTERVIEW" in p["passed_stages"]),
                "passed_technical": sum(1 for p in progress_data if "TECHNICAL_ASSESSMENT" in p["passed_stages"]),
                "passed_client_interview": sum(1 for p in progress_data if "CLIENT_INTERVIEW" in p["passed_stages"]),
                "received_offers": sum(1 for p in progress_data if p["current_stage"] == "FINAL_EVALUATION" or p["current_status"] in ["OFFER_EXTENDED", "OFFER_ACCEPTED", "HIRED", "JOINED"]),
                "failed_applications": sum(1 for p in progress_data if p["is_failed"]),
                "stage_breakdown": stage_counts,
            }

            return json.dumps({
                "applications": progress_data,
                "summary": summary,
            }, default=str)

    except Exception as e:
        logger.error(f"Error in get_candidate_pipeline_progress: {e}")
        return json.dumps({"error": str(e)})


# =======================================================================
#                    RECRUITER CANDIDATE ANALYTICS TOOLS
# =======================================================================
# These tools provide analytics about candidates for recruiters.
# All tools use ORM queries - no hardcoded schemas.


@mcp.tool(
    description=(
        "Get comprehensive pipeline analytics for a job showing candidate distribution "
        "across all stages, conversion rates between stages, and detailed statistics. "
        "Use format: job_id (UUID)"
    )
)
async def get_candidate_pipeline_analytics(job_id: str) -> str:
    """
    Get pipeline analytics with candidate distribution and conversion rates.
    Uses existing PipelineService.get_pipeline_statistics() - no raw SQL.
    """
    logger.info(f"Tool 'get_candidate_pipeline_analytics' called for job: {job_id}")

    try:
        job_uuid = uuid.UUID(job_id.strip())

        async with AsyncSessionLocal() as session:
            # Get pipeline statistics using existing service method
            stats = await PipelineService.get_pipeline_statistics(
                db=session, job_posting_id=job_uuid
            )

            # Get job title
            job_result = await session.execute(
                select(JobPosting.title).where(JobPosting.job_posting_id == job_uuid)
            )
            job_title = job_result.scalar() or "Unknown Job"

            # Calculate conversion rates from status breakdown
            status_breakdown = stats.get("status_breakdown", {})
            total = sum(status_breakdown.values())

            conversion_rates = {}
            if total > 0:
                sourced = status_breakdown.get("SOURCED", 0)
                ranked = status_breakdown.get("RANKED", 0)
                screening_passed = status_breakdown.get("SCREENING_PASSED", 0)

                if sourced > 0:
                    conversion_rates["sourced_to_ranked"] = round((ranked / sourced) * 100, 1)
                    conversion_rates["sourced_to_screening_passed"] = round((screening_passed / sourced) * 100, 1)

                if screening_passed > 0:
                    conversion_rates["screening_to_ranked"] = round((ranked / screening_passed) * 100, 1)

            return json.dumps({
                "job_id": str(job_uuid),
                "job_title": job_title,
                "total_candidates": total,
                "status_breakdown": status_breakdown,
                "conversion_rates": conversion_rates,
                "stage_statistics": stats.get("stage_statistics", []),
            }, indent=2, default=str)

    except ValueError as e:
        logger.error(f"Invalid job_id format: {e}")
        return json.dumps({"error": f"Invalid job_id format: {str(e)}"})
    except Exception as e:
        logger.error(f"Error in get_candidate_pipeline_analytics: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description=(
        "Get application status breakdown with counts and percentages. "
        "Shows distribution of candidates across all statuses. "
        "Use format: job_id (UUID, optional - if omitted, shows global stats)"
    )
)
async def get_candidate_status_breakdown(job_id: str = "") -> str:
    """
    Get detailed status breakdown with counts and percentages.
    Uses ORM queries via AnalyticsService - no hardcoded schema.
    """
    logger.info(f"Tool 'get_candidate_status_breakdown' called for job: {job_id or 'all'}")

    try:
        job_uuid = uuid.UUID(job_id.strip()) if job_id.strip() else None

        async with AsyncSessionLocal() as session:
            result = await AnalyticsService.get_application_status_breakdown(
                db=session, job_id=job_uuid
            )
            return json.dumps(result, indent=2, default=str)

    except ValueError as e:
        logger.error(f"Invalid job_id format: {e}")
        return json.dumps({"error": f"Invalid job_id format: {str(e)}"})
    except Exception as e:
        logger.error(f"Error in get_candidate_status_breakdown: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description=(
        "Get skill distribution across candidates showing top skills, "
        "skill types breakdown, and average experience per skill. "
        "Use format: job_id|limit (e.g., 'uuid|20' or just 'uuid' for default 20 skills, "
        "or empty for global stats)"
    )
)
async def get_candidate_skill_distribution(input_data: str = "") -> str:
    """
    Analyze skill distribution across candidates.
    Uses ORM queries via AnalyticsService - no hardcoded schema.
    """
    logger.info(f"Tool 'get_candidate_skill_distribution' called with: {input_data or 'global'}")

    try:
        # Parse input
        parts = input_data.split("|") if input_data else [""]
        job_id_str = parts[0].strip() if parts else ""
        limit = int(parts[1].strip()) if len(parts) > 1 else 20

        job_uuid = uuid.UUID(job_id_str) if job_id_str else None

        async with AsyncSessionLocal() as session:
            result = await AnalyticsService.get_skill_distribution(
                db=session, job_id=job_uuid, limit=limit
            )
            return json.dumps(result, indent=2, default=str)

    except ValueError as e:
        logger.error(f"Invalid input format: {e}")
        return json.dumps({"error": f"Invalid input format: {str(e)}"})
    except Exception as e:
        logger.error(f"Error in get_candidate_skill_distribution: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description=(
        "Get experience distribution analytics showing experience ranges, "
        "average experience, domain distribution, and location distribution. "
        "Use format: job_id (UUID, optional - if omitted, shows global stats)"
    )
)
async def get_candidate_experience_analytics(job_id: str = "") -> str:
    """
    Analyze experience distribution across candidates.
    Uses ORM queries via AnalyticsService - no hardcoded schema.
    """
    logger.info(f"Tool 'get_candidate_experience_analytics' called for job: {job_id or 'all'}")

    try:
        job_uuid = uuid.UUID(job_id.strip()) if job_id.strip() else None

        async with AsyncSessionLocal() as session:
            result = await AnalyticsService.get_experience_distribution(
                db=session, job_id=job_uuid
            )
            return json.dumps(result, indent=2, default=str)

    except ValueError as e:
        logger.error(f"Invalid job_id format: {e}")
        return json.dumps({"error": f"Invalid job_id format: {str(e)}"})
    except Exception as e:
        logger.error(f"Error in get_candidate_experience_analytics: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description=(
        "Get sourcing analytics for a job including sourcing methods, "
        "fitment score distribution, and progression rates from sourcing. "
        "Use format: job_id (UUID, required)"
    )
)
async def get_candidate_sourcing_analytics(job_id: str) -> str:
    """
    Analyze sourcing effectiveness for a job.
    Uses ORM queries via AnalyticsService - no hardcoded schema.
    """
    logger.info(f"Tool 'get_candidate_sourcing_analytics' called for job: {job_id}")

    try:
        job_uuid = uuid.UUID(job_id.strip())

        async with AsyncSessionLocal() as session:
            result = await AnalyticsService.get_sourcing_analytics(
                db=session, job_id=job_uuid
            )
            return json.dumps(result, indent=2, default=str)

    except ValueError as e:
        logger.error(f"Invalid job_id format: {e}")
        return json.dumps({"error": f"Invalid job_id format: {str(e)}"})
    except Exception as e:
        logger.error(f"Error in get_candidate_sourcing_analytics: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description=(
        "Get ranking analytics for a job showing GPT score distribution, "
        "average scores, and recommendation breakdown. "
        "Use format: job_id (UUID, required)"
    )
)
async def get_candidate_ranking_analytics(job_id: str) -> str:
    """
    Analyze ranking performance for a job.
    Uses RankedCandidateService.get_ranked_candidates() for data retrieval.
    """
    logger.info(f"Tool 'get_candidate_ranking_analytics' called for job: {job_id}")

    try:
        job_uuid = uuid.UUID(job_id.strip())

        async with AsyncSessionLocal() as session:
            from app.candidates.services.ranking_service import RankedCandidateService

            service = RankedCandidateService(session)
            result = await service.get_ranked_candidates(job_uuid)

            # Analyze ranking scores
            scores = []
            recommendations = {"strong_fit": 0, "good_fit": 0, "moderate_fit": 0, "weak_fit": 0}

            candidates = result.candidates if hasattr(result, 'candidates') else []
            for candidate in candidates:
                score = candidate.final_score if hasattr(candidate, 'final_score') else None
                if score is not None:
                    scores.append(float(score))
                    # Categorize by score
                    if score >= 80:
                        recommendations["strong_fit"] += 1
                    elif score >= 60:
                        recommendations["good_fit"] += 1
                    elif score >= 40:
                        recommendations["moderate_fit"] += 1
                    else:
                        recommendations["weak_fit"] += 1

            # Calculate score ranges
            score_ranges = {"0-25": 0, "26-50": 0, "51-75": 0, "76-100": 0}
            for score in scores:
                if score <= 25:
                    score_ranges["0-25"] += 1
                elif score <= 50:
                    score_ranges["26-50"] += 1
                elif score <= 75:
                    score_ranges["51-75"] += 1
                else:
                    score_ranges["76-100"] += 1

            avg_score = sum(scores) / len(scores) if scores else 0.0

            return json.dumps({
                "job_id": str(job_uuid),
                "total_ranked": len(candidates),
                "score_ranges": score_ranges,
                "average_score": round(avg_score, 1),
                "recommendation_breakdown": recommendations,
                "scores_available": len(scores),
            }, indent=2, default=str)

    except ValueError as e:
        logger.error(f"Invalid job_id format: {e}")
        return json.dumps({"error": f"Invalid job_id format: {str(e)}"})
    except Exception as e:
        logger.error(f"Error in get_candidate_ranking_analytics: {e}")
        return json.dumps({"error": str(e)})


# =======================================================================
#                    Simplified Pipeline Tools (using basic types only)
# =======================================================================


@mcp.tool(
    description=(
        "Moves selected candidates to a different pipeline stage. "
        "Format: job_id|candidate_id1,candidate_id2|target_stage|skip_validation|execute_stage_logic|notes\n\n"
        "Available stages:\n"
        "- SOURCING\n- RANKING\n- PRESCREENING\n- AI_INTERVIEW\n"
        "- AI_VIDEO_INTERVIEW\n- TECHNICAL_ASSESSMENT\n"
        "- CLIENT_INTERVIEW\n- FINAL_EVALUATION\n\n"
        "skip_validation: 'true' to bypass validation, 'false' to enforce (default: false)\n"
        "execute_stage_logic: 'true' to run intermediate stage logic, 'false' to skip (default: true)\n"
        "notes: Optional notes about the movement"
    )
)
async def move_candidates_to_stage_simple(input_data: str) -> Dict[str, Any]:
    """Move candidates to a specific pipeline stage using simple string input"""
    logger.info(f"Tool 'move_candidates_to_stage_simple' called with: {input_data}")

    try:
        # Parse the pipe-separated input
        parts = input_data.split("|")
        if len(parts) < 3:
            raise ValueError(
                "Input must have at least job_id|candidate_ids|target_stage"
            )

        job_id_str = parts[0].strip()
        candidate_ids_str = parts[1].strip()
        target_stage = parts[2].strip()

        # Robust skip_validation parsing - accepts multiple formats
        skip_validation_str = parts[3].strip().lower() if len(parts) > 3 else "false"
        skip_validation = skip_validation_str in ["true", "yes", "1", "t", "y"]

        # NEW: Execute stage logic parameter - controls intermediate stage execution
        execute_stage_logic_str = parts[4].strip().lower() if len(parts) > 4 else "true"
        execute_stage_logic = execute_stage_logic_str in ["true", "yes", "1", "t", "y"]

        # Notes (now 6th parameter)
        notes = parts[5].strip() if len(parts) > 5 else None

        # Parse candidate IDs
        candidate_ids = [cid.strip() for cid in candidate_ids_str.split(",")]

        # Import types at runtime
        from app.jobs.schemas.pipeline_schemas import (
            MoveCandidatesRequest,
            CandidateStageMovement,
        )
        from app.db.schemas import ProcessType

        # Create movement object
        movement = CandidateStageMovement(
            candidate_ids=[uuid.UUID(cid) for cid in candidate_ids],
            target_process_type=ProcessType(target_stage),
            skip_validation=skip_validation,
            notes=notes,
        )

        # Create request
        request = MoveCandidatesRequest(
            job_posting_id=uuid.UUID(job_id_str), movement=movement
        )

        async with AsyncSessionLocal() as session:
            result = await PipelineService.move_candidates_to_stage(
                db=session,
                request=request,
                execute_stage_logic=execute_stage_logic  # NEW: Pass execute_stage_logic parameter
            )
            return {
                "moved_count": result.moved_count,
                "skipped_count": result.skipped_count,
                "message": result.message,
                "failed_candidates": result.failed_candidates,
            }

    except Exception as e:
        logger.error(f"Error in move_candidates_to_stage_simple: {e}")
        return {"error": str(e), "moved_count": 0}


@mcp.tool(
    description="Gets the default pipeline template that can be used for job configuration. Returns template as JSON string."
)
async def get_default_pipeline_template_simple() -> str:
    """Get the default pipeline template as JSON string"""
    logger.info("Tool 'get_default_pipeline_template_simple' called")

    try:
        template = await PipelineService.get_default_pipeline_template()

        # Convert to basic dict format
        template_dict = {
            "name": template.name,
            "description": template.description,
            "steps": [
                {
                    "process_type": step.process_type.value,
                    "process_order": step.process_order,
                    "is_active": step.is_active,
                    "is_required": step.is_required,
                    "config": step.config,
                }
                for step in template.steps
            ],
        }

        return json.dumps(template_dict, indent=2)

    except Exception as e:
        logger.error(f"Error in get_default_pipeline_template_simple: {e}")
        return f'{{"error": "{str(e)}"}}'


@mcp.tool(
    description="Creates default 6-stage pipeline for an existing job. Use format: job_id"
)
async def create_default_pipeline_for_job(job_id: str) -> str:
    """Create default pipeline for an existing job"""
    logger.info(f"Tool 'create_default_pipeline_for_job' called for job: {job_id}")

    try:
        # Import types at runtime
        from app.jobs.schemas.pipeline_schemas import CreatePipelineRequest

        # Get default template
        template = await PipelineService.get_default_pipeline_template()

        # Create request
        request = CreatePipelineRequest(
            job_posting_id=uuid.UUID(job_id), pipeline_template=template
        )

        async with AsyncSessionLocal() as session:
            result = await PipelineService.create_pipeline_for_job(
                db=session, request=request
            )
            return f"Successfully created default pipeline with {len(result.processes_created)} stages for job {job_id}"

    except Exception as e:
        logger.error(f"Error in create_default_pipeline_for_job: {e}")
        return f"Error creating pipeline: {str(e)}"


# =========================================================================
#                   CLIENT TOOLS
# =========================================================================
@mcp.tool(
    description=(
        "Parse a job request and create a job requirement directly with AI-generated JD. "
        "User must provide: client_id, location, industry, openings, SLA days, priority, "
        "employment type, experience (min years), and skills. "
        "This creates the job requirement in ONE step (no draft)."
    )
)
async def client_create_job_requirement_direct(input_data: str) -> str:
    """
    NEW: Direct job requirement creation for clients (no draft).
    Format: raw_text|client_id|location|industry|openings|sla_days|priority|employment_type|user_id

    Example: "Python developer with 5 years|uuid1|NYC|Tech|2|30|HIGH|FULL_TIME|uuid2"
    """
    logger.info("Tool 'client_create_job_requirement_direct' called - direct creation.")

    try:
        # Parse pipe-separated input
        parts = input_data.split("|")
        if len(parts) < 9:
            return json.dumps(
                {
                    "error": "Invalid format. Use: raw_text|client_id|location|industry|openings|sla_days|priority|employment_type|user_id"
                }
            )

        raw_text = parts[0].strip()
        client_id = uuid.UUID(parts[1].strip())
        location = parts[2].strip() if parts[2].strip() else None
        industry = parts[3].strip() if parts[3].strip() else None
        openings = int(parts[4].strip())
        sla_days = int(parts[5].strip()) if parts[5].strip() else None
        priority = parts[6].strip().upper()
        employment_type = parts[7].strip().upper()
        user_id = uuid.UUID(parts[8].strip())

        # Parse job request
        parsed_result = await JobParsingService.parse_job_request_detailed(
            JobRequestParseInput(raw_text=raw_text)
        )
        logger.info(
            f"Parsed job: {parsed_result.title} with {len(parsed_result.skills)} skills"
        )

        # Generate JD directly (no draft)
        jd_input = GenerateJDInput(
            title=parsed_result.title,
            skills=parsed_result.skills,
            location=location or parsed_result.location,
            experience=parsed_result.experience,
            max_experience=parsed_result.max_experience,
            domain=parsed_result.domain,
            number_of_positions=parsed_result.number_of_positions,
            time_to_fill_days=parsed_result.time_to_fill_days,
            priority=parsed_result.priority,
            job_requirement_id=input_data.job_requirement_id,
        )

        (
            structured_jd,
            job_description_full_text,
        ) = await JobDescriptionService.generate_jd_for_requirement(jd_input)

        # Create job requirement directly
        async with AsyncSessionLocal() as session:
            job_requirement = await JobRequirementService.create_with_ai_jd(
                db=session,
                client_id=client_id,
                user_id=user_id,
                created_by_role="CLIENT_USER",
                title=parsed_result.title,
                location=location or parsed_result.location,
                industry=industry,
                openings=openings,
                sla_days=sla_days,
                priority=priority,
                employment_type=employment_type,
                experience=parsed_result.experience,
                max_experience=parsed_result.max_experience,
                skills=[skill.model_dump() for skill in parsed_result.skills],
                structured_jd=structured_jd.model_dump(),
                job_description_full_text=job_description_full_text,
            )

            logger.success(
                f"Created job requirement {job_requirement.job_requirement_id} directly"
            )

            return json.dumps(
                {
                    "job_requirement_id": str(job_requirement.job_requirement_id),
                    "title": job_requirement.title,
                    "message": "Job requirement created successfully!",
                    "status": "success",
                },
                default=str,
            )

    except Exception as e:
        logger.error(f"Error in client_create_job_requirement_direct: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description=(
        "Creates a job requirement after the JD has been approved. "
        "This is step 2 - collects client name, job details like "
        "location, industry, openings, SLA, priority, employment type. "
        "Use the approved draft_id from step 1."
    )
)
async def create_job_requirement_from_jd_draft(
    input_data: CreateJobRequirementInput,
) -> CreateJobRequirementResponse:
    """
    Step 2: Create job requirement with all business details.
    This is called AFTER the client reviews and approves the JD.
    """
    logger.info(
        f"Tool 'create_job_requirement_from_jd_draft' called for "
        f"draft: {input_data.draft_id}"
    )

    async with AsyncSessionLocal() as session:
        # Get the job draft with all details
        stmt = select(JobDraft).where(JobDraft.draft_id == input_data.draft_id)
        result = await session.execute(stmt)
        job_draft = result.scalar_one_or_none()

        if not job_draft:
            logger.error(f"Could not find job draft with ID: {input_data.draft_id}")
            raise ValueError(f"Draft with ID {input_data.draft_id} not found.")

        # Extract JD details from draft
        jd_details = job_draft.structured_jd

        # Convert priority string to enum (REQUIRED)
        from app.db.schemas import JobPriority
        if not input_data.priority:
            raise ValueError("Priority is required and cannot be None")

        try:
            priority_enum = JobPriority[input_data.priority.upper()]
            logger.info(f"✓ Priority validated: {input_data.priority} → {priority_enum.value}")
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Invalid priority '{input_data.priority}'. Must be HIGH, MEDIUM, or LOW") from e

        # Create the job requirement with all collected details
        new_job_requirement = JobRequirement(
            client_id=input_data.client_id,
            title=input_data.title,
            location=input_data.location,
            industry=input_data.industry,
            openings=input_data.openings,
            sla_days=input_data.sla_days,
            priority=priority_enum,
            employment_type=input_data.employment_type,  # Direct string: FULL_TIME, etc.
            status="OPEN",  # Initial status
            job_requirement_details=jd_details,  # Store full JD from draft
        )

        session.add(new_job_requirement)
        await session.commit()
        await session.refresh(new_job_requirement)

        logger.success(
            f"Successfully created job requirement "
            f"{new_job_requirement.job_requirement_id}"
        )

        return CreateJobRequirementResponse(
            job_requirement_id=new_job_requirement.job_requirement_id,
            message=(
                "Job requirement created successfully. "
                "You can now proceed with assigning recruiters or "
                "setting up interview rounds."
            ),
            status="success",
        )


# =============================================================================
# TECHNICAL ASSESSMENT CONFIGURATION TOOLS
# =============================================================================


@mcp.tool()
async def get_available_technical_assessments(arguments: dict) -> dict:
    """
    Fetch available Xobin technical assessments for configuration.

    This tool retrieves all available assessments from Xobin that can be
    configured for the technical assessment stage of a job posting.

    Args:
        arguments: Empty dict (no arguments required)

    Returns:
        Dict containing:
        - success (bool): Operation status
        - assessments (list): List of available assessments with id, name, and details
        - count (int): Number of assessments available

    Example Response:
    {
        "success": true,
        "assessments": [
            {
                "assessment_id": 123,
                "assessment_name": "Python Developer Assessment",
                "created_by": "admin@xobin.com"
            }
        ],
        "count": 10
    }
    """
    async with AsyncSessionLocal() as db:
        from app.technical_interview_xobin.services.technical_assessment_config_service import (
            TechnicalAssessmentConfigService,
        )

        try:
            assessments = (
                await TechnicalAssessmentConfigService.fetch_available_assessments(db)
            )
            return {
                "success": True,
                "assessments": assessments,
                "count": len(assessments),
            }
        except Exception as e:
            logger.error(f"Error fetching technical assessments: {e}")
            return {"success": False, "error": str(e), "assessments": [], "count": 0}


@mcp.tool()
async def get_technical_assessment_details(arguments: dict) -> dict:
    """
    Get detailed information about a specific Xobin assessment.

    This tool fetches comprehensive details about a specific assessment,
    including questions, duration, topics, and difficulty level.

    Args:
        arguments: Dict containing:
            - assessment_id (int): Xobin assessment ID

    Returns:
        Dict containing:
        - success (bool): Operation status
        - assessment_details (dict): Complete assessment information

    Example Usage:
    {
        "assessment_id": 123
    }

    Example Response:
    {
        "success": true,
        "assessment_details": {
            "assessment_id": 123,
            "assessment_name": "Python Developer Assessment",
            "duration": 60,
            "question_count": 25,
            "topics": ["Python Basics", "Data Structures", "Algorithms"]
        }
    }
    """
    assessment_id = arguments.get("assessment_id")

    if not assessment_id:
        return {"success": False, "error": "assessment_id is required"}

    async with AsyncSessionLocal() as db:
        from app.technical_interview_xobin.services.technical_assessment_config_service import (
            TechnicalAssessmentConfigService,
        )

        try:
            details = await TechnicalAssessmentConfigService.get_assessment_details(
                db, assessment_id
            )
            return {"success": True, "assessment_details": details}
        except Exception as e:
            logger.error(f"Error fetching assessment details for {assessment_id}: {e}")
            return {"success": False, "error": str(e)}


@mcp.tool()
async def configure_technical_assessment_stage(arguments: dict) -> dict:
    """
    Configure technical assessment stage for a job draft.

    This tool configures the technical assessment stage in the pipeline draft
    by selecting a Xobin assessment, setting link expiry, and configuring proctoring.

    Args:
        arguments: Dict containing:
            - job_draft_id (str): UUID of the job draft
            - assessment_id (int): Xobin assessment ID
            - assessment_name (str): Name of the assessment
            - expiry_in_days (int, optional): Link expiry (1-10 days, default 5)
            - proctoring_ai (bool, optional): Enable AI proctoring (default true)
            - proctoring_eyegaze (bool, optional): Enable eye-gaze tracking (default true)
            - proctoring_offtab (bool, optional): Enable tab detection (default true)
            - proctoring_screen_record (bool, optional): Enable screen recording (default true)

    Returns:
        Dict containing:
        - success (bool): Operation status
        - message (str): Success/error message
        - config (dict): Applied configuration

    Example Usage:
    {
        "job_draft_id": "123e4567-e89b-12d3-a456-426614174000",
        "assessment_id": 123,
        "assessment_name": "Python Developer Assessment",
        "expiry_in_days": 5,
        "proctoring_ai": true,
        "proctoring_eyegaze": true,
        "proctoring_offtab": true,
        "proctoring_screen_record": false
    }
    """
    job_draft_id_str = arguments.get("job_draft_id")
    assessment_id = arguments.get("assessment_id")
    assessment_name = arguments.get("assessment_name")

    if not job_draft_id_str or not assessment_id or not assessment_name:
        return {
            "success": False,
            "error": "job_draft_id, assessment_id, and assessment_name are required",
        }

    try:
        from uuid import UUID

        job_draft_id = UUID(job_draft_id_str)
    except ValueError:
        return {"success": False, "error": "Invalid job_draft_id format"}

    config = {
        "assessment_provider": "xobin",
        "chosen_assessment": {
            "assessment_id": assessment_id,
            "assessment_name": assessment_name,
        },
        "expiry_in_days": arguments.get("expiry_in_days", 5),
        "proctoring_settings": {
            "ai": arguments.get("proctoring_ai", True),
            "eyegaze": arguments.get("proctoring_eyegaze", True),
            "offtab": arguments.get("proctoring_offtab", True),
            "screen_record": arguments.get("proctoring_screen_record", True),
        },
    }

    async with AsyncSessionLocal() as db:
        from app.jobs.services.pipeline_service import PipelineService

        try:
            result = await PipelineService.configure_technical_assessment_stage_draft(
                db, job_draft_id, config
            )
            return {
                "success": True,
                "message": "Technical assessment configured successfully",
                "config": result,
            }
        except Exception as e:
            logger.error(f"Error configuring technical assessment: {e}")
            return {"success": False, "error": str(e)}


# =============================================================================
# JOB REQUIREMENT MANAGEMENT TOOLS
# =============================================================================


@mcp.tool(
    description="Get all job requirements assigned to the current recruiter. Returns list of jobs with title, client, status, priority, and other details. Use recruiter_id parameter."
)
@ensure_json_response
async def get_assigned_job_requirements(recruiter_id: str) -> str:
    """
    Get all job requirements assigned to a recruiter.

    This tool retrieves all jobs that have been assigned to the recruiter,
    including the job requirement details, client information, and draft status.

    Args:
        recruiter_id: UUID string of the recruiter

    Returns:
        JSON string with list of assigned job requirements
    """
    logger.info(
        f"Tool 'get_assigned_job_requirements' called for recruiter: {recruiter_id}"
    )

    try:
        async with AsyncSessionLocal() as session:
            from app.jobs.services.job_requirement_service import JobRequirementService

            requirements = (
                await JobRequirementService.get_assigned_requirements_for_recruiter(
                    db=session, recruiter_id=uuid.UUID(recruiter_id)
                )
            )

            if not requirements:
                return json.dumps(
                    {
                        "count": 0,
                        "requirements": [],
                        "message": "No job requirements assigned to this recruiter yet.",
                    },
                    indent=2,
                )

            return json.dumps(
                {
                    "count": len(requirements),
                    "requirements": requirements,
                    "message": f"Found {len(requirements)} assigned job requirement(s).",
                },
                indent=2,
            )

    except ValueError as e:
        logger.error(f"Invalid UUID in get_assigned_job_requirements: {e}")
        return json.dumps({"error": "Invalid recruiter_id format"})
    except Exception as e:
        logger.error(f"Error in get_assigned_job_requirements: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description="Get clients for a recruiter based on organizational hierarchy (Recruiter → RecruiterLead → Sales → Clients). Returns clients from the recruiter's sales organization, or all clients if no RecruiterLead assigned."
)
@ensure_json_response
async def list_recruiter_clients(recruiter_id: str) -> str:
    """
    Get clients for a specific recruiter using organizational hierarchy.

    This tool filters clients based on the hierarchy:
    Recruiter → RecruiterLead → Sales → Clients

    If recruiter has a RecruiterLead → returns clients from their Sales organization
    If recruiter has NO RecruiterLead → returns ALL clients (fallback)

    Args:
        recruiter_id: UUID string of the recruiter

    Returns:
        JSON string with:
        - count: Number of clients found
        - clients: List of client objects (client_id, client_name)
        - message: Status message
        - filtered_by_sales: Boolean indicating if filtering was applied
        - sales_id: Sales organization ID (if filtering applied)
    """
    logger.info(f"Tool 'list_recruiter_clients' called for recruiter: {recruiter_id}")

    try:
        async with AsyncSessionLocal() as session:
            from sqlalchemy.orm import selectinload

            # Convert recruiter_id to UUID
            try:
                recruiter_uuid = uuid.UUID(recruiter_id)
            except ValueError:
                return json.dumps({"error": "Invalid recruiter_id format"}, indent=2)

            # Get recruiter with relationships loaded
            recruiter_stmt = (
                select(Recruiter)
                .options(
                    selectinload(Recruiter.recruiter_lead)
                    .selectinload(RecruiterLead.sales)
                    .selectinload(Sales.clients)
                    .selectinload(Client.user),
                    selectinload(Recruiter.recruiter_lead)
                    .selectinload(RecruiterLead.sales)
                    .selectinload(Sales.clients)
                    .selectinload(Client.company)
                )
                .where(Recruiter.recruiter_id == recruiter_uuid)
            )

            recruiter_result = await session.execute(recruiter_stmt)
            recruiter = recruiter_result.scalar_one_or_none()

            if not recruiter:
                return json.dumps({
                    "error": f"Recruiter {recruiter_id} not found"
                }, indent=2)

            # Determine which clients to return based on hierarchy
            if recruiter.recruiter_lead and recruiter.recruiter_lead.sales:
                # Return clients from the recruiter's sales organization
                clients = recruiter.recruiter_lead.sales.clients
                filter_applied = True
                sales_id = str(recruiter.recruiter_lead.sales.sales_id)
                logger.info(f"Filtering clients by sales_id: {sales_id}")
            else:
                # Fallback: return all clients if no recruiter lead assigned
                logger.info(f"Recruiter {recruiter_id} has no RecruiterLead - returning all clients")
                all_clients_stmt = (
                    select(Client)
                    .options(
                        selectinload(Client.user),
                        selectinload(Client.company)
                    )
                )
                all_clients_result = await session.execute(all_clients_stmt)
                clients = all_clients_result.scalars().all()
                filter_applied = False
                sales_id = None

            if not clients:
                return json.dumps({
                    "count": 0,
                    "clients": [],
                    "message": "No clients found.",
                    "filtered_by_sales": filter_applied,
                    "sales_id": sales_id
                }, indent=2)

            # Build client list
            clients_data = []
            for client in clients:
                # Format: "company name - client name" or "N/A - client name"
                company_name = client.company.company_name if client.company else "N/A"
                client_name = client.user.full_name if client.user else "N/A"
                display_name = f"{company_name} - {client_name}"

                client_info = {
                    "client_id": str(client.client_id),
                    "client_name": display_name
                }
                clients_data.append(client_info)

            return json.dumps({
                "count": len(clients_data),
                "clients": clients_data,
                "message": f"Found {len(clients_data)} client(s) in the database.",
                "filtered_by_sales": filter_applied,
                "sales_id": sales_id
            }, indent=2)

    except ValueError as e:
        logger.error(f"Invalid UUID in list_recruiter_clients: {e}")
        return json.dumps({"error": "Invalid recruiter_id format"})
    except Exception as e:
        logger.error(f"Error in list_recruiter_clients: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description="Verify if a client exists in CEIPAL. Call immediately after user selects a client. Returns status without creating anything. Always verifies with CEIPAL API to ensure data is current."
)
@ensure_json_response
async def verify_client_in_ceipal(client_id: str) -> str:
    """
    Verify if a client exists in CEIPAL system.

    This tool checks if a client is registered in CEIPAL by looking up their email.
    It always validates with the CEIPAL API, even if a cached ID exists, to ensure accuracy.

    Args:
        client_id: UUID string of the client to verify

    Returns:
        JSON string with:
        - client_id: The client UUID
        - client_name: Company name
        - client_email: Client's email address
        - exists_in_ceipal: Boolean indicating if client found in CEIPAL
        - ceipal_client_id: Encoded CEIPAL ID if found, null otherwise
        - cached_id: Boolean indicating if ID was previously cached
        - cache_status: Status of cache ("cached_valid", "cached_stale", "not_cached")
        - message: User-friendly status message
    """
    logger.info(f"Tool 'verify_client_in_ceipal' called for client_id: {client_id}")

    try:
        async with AsyncSessionLocal() as db:
            from sqlalchemy.orm import selectinload
            from app.ceipal.services.ceipal_authentication import CEIPAL
            from app.ceipal.schemas import CeipalAPIEndpoint

            # Convert client_id to UUID
            try:
                client_uuid = uuid.UUID(client_id)
            except ValueError:
                return json.dumps({"error": "Invalid client_id format"}, indent=2)

            # Query database for Client with Company and User info
            stmt = (
                select(Client)
                .options(
                    selectinload(Client.company),
                    selectinload(Client.user)
                )
                .where(Client.client_id == client_uuid)
            )
            result = await db.execute(stmt)
            client = result.scalar_one_or_none()

            if not client:
                return json.dumps({"error": f"Client {client_id} not found in database"}, indent=2)

            # Extract client details
            company_name = client.company.company_name if client.company else "Unknown"
            client_email = client.user.email if client.user else None

            if not client_email:
                return json.dumps({"error": "Client has no email address"}, indent=2)

            # Check if client has cached CEIPAL ID
            cached_id = None
            had_cache = False
            if client.client_details and isinstance(client.client_details, dict):
                cached_id = client.client_details.get("ceipal_client_contact_id")
                had_cache = cached_id is not None

            # ALWAYS verify with CEIPAL API (don't trust cache blindly)
            logger.info(f"Verifying client '{company_name}' ({client_email}) in CEIPAL...")

            ceipal = CEIPAL(
                email=settings.ceipal_email,
                password=settings.ceipal_password,
                api_key=settings.ceipal_api_key,
                db=db,
            )

            try:
                lookup = await ceipal.lookup_email_in_api(
                    client_email,
                    CeipalAPIEndpoint.CLIENT_CONTACTS
                )
            except Exception as ceipal_error:
                # CEIPAL API error - return error with retry suggestion
                logger.error(f"CEIPAL API error during verification: {ceipal_error}")
                return json.dumps({
                    "client_id": client_id,
                    "client_name": company_name,
                    "client_email": client_email,
                    "exists_in_ceipal": None,
                    "error": "CEIPAL_API_UNAVAILABLE",
                    "message": f"CEIPAL API temporarily unavailable: {str(ceipal_error)}. Please retry.",
                    "retry_recommended": True
                }, indent=2)

            # Check if client exists in CEIPAL
            if lookup.get("found") and lookup.get("id"):
                ceipal_id = lookup["id"]

                # Determine cache status
                if not had_cache:
                    cache_status = "not_cached"
                    logger.info(f"Client found in CEIPAL (ID: {ceipal_id}). No cache existed.")
                elif cached_id == ceipal_id:
                    cache_status = "cached_valid"
                    logger.success(f"Client found in CEIPAL. Cache is valid (ID: {ceipal_id}).")
                else:
                    cache_status = "cached_stale"
                    logger.warning(
                        f"Client found in CEIPAL but cache is stale. "
                        f"Cached: {cached_id}, CEIPAL: {ceipal_id}. Updating cache..."
                    )
                    # Update cache with correct ID
                    client.client_details = client.client_details or {}
                    client.client_details["ceipal_client_contact_id"] = ceipal_id
                    client.client_details["ceipal_verified_at"] = datetime.now(UTC).isoformat()
                    await db.commit()

                return json.dumps({
                    "client_id": client_id,
                    "client_name": company_name,
                    "client_email": client_email,
                    "exists_in_ceipal": True,
                    "ceipal_client_id": ceipal_id,
                    "cached_id": had_cache,
                    "cache_status": cache_status,
                    "message": f"Client '{company_name}' is registered in CEIPAL."
                }, indent=2)
            else:
                # Client NOT found in CEIPAL
                logger.warning(f"Client '{company_name}' ({client_email}) not found in CEIPAL.")

                # Clear stale cache if it exists
                if had_cache:
                    logger.info("Clearing stale cache...")
                    if client.client_details:
                        client.client_details.pop("ceipal_client_id", None)
                        await db.commit()

                return json.dumps({
                    "client_id": client_id,
                    "client_name": company_name,
                    "client_email": client_email,
                    "exists_in_ceipal": False,
                    "ceipal_client_id": None,
                    "cached_id": False,
                    "cache_status": "not_in_ceipal",
                    "message": f"Client '{company_name}' is not registered in CEIPAL. Use create_ceipal_client to register them."
                }, indent=2)

    except Exception as e:
        logger.error(f"Error in verify_client_in_ceipal: {e}")
        return json.dumps({"error": f"Error verifying client: {str(e)}"}, indent=2)


@mcp.tool(
    description="IMPORTANT: This tool ONLY registers clients that already exist in our database into CEIPAL. It does NOT create new clients from scratch. Only call after verify_client_in_ceipal returns 'not found' for a client that EXISTS in your client list. Creates both client and client contact in CEIPAL and caches both IDs. Validates that the client belongs to the recruiter's sales organization."
)
@ensure_json_response
async def create_ceipal_client(client_id: str, recruiter_id: str) -> str:
    """
    Create a client in CEIPAL using the two-step process and cache both IDs.

    Validates that the client belongs to the recruiter's sales organization before creating.

    Step 1: Create client in saveCustomClientsDetails (returns client_enc_id)
    Step 2: Create client contact in saveCustomClientContactsDetails (returns client_contact_enc_id)

    Error Handling:
    - If client not in recruiter's sales org: Return error, nothing cached
    - If Step 1 fails: Return error, nothing cached
    - If Step 1 succeeds but Step 2 fails: Retry Step 2 once
    - If Step 2 still fails: "Forget" Step 1 (don't cache anything), return error

    Args:
        client_id: UUID string of the client to create in CEIPAL
        recruiter_id: UUID string of the recruiter requesting the creation

    Returns:
        JSON string with:
        - success: Boolean indicating if BOTH steps succeeded
        - client_id: The client UUID
        - client_name: Company name
        - client_email: Client's email address
        - ceipal_client_id: Client ID from Step 1
        - ceipal_client_contact_id: Client contact ID from Step 2
        - message: Success/error message
        - cached: Boolean indicating if IDs were cached in database
    """
    logger.info(f"Tool 'create_ceipal_client' called for client_id: {client_id}")

    try:
        async with AsyncSessionLocal() as db:
            from sqlalchemy.orm import selectinload
            from app.ceipal.services.ceipal_authentication import CEIPAL
            from app.db.models import Sales

            # Convert client_id to UUID
            try:
                client_uuid = uuid.UUID(client_id)
            except ValueError:
                return json.dumps({"error": "Invalid client_id format", "success": False}, indent=2)

            # Convert recruiter_id to UUID
            try:
                recruiter_uuid = uuid.UUID(recruiter_id)
            except ValueError:
                return json.dumps({"error": "Invalid recruiter_id format", "success": False}, indent=2)

            # Validate recruiter and get their sales organization
            recruiter_stmt = (
                select(Recruiter)
                .options(
                    selectinload(Recruiter.recruiter_lead)
                    .selectinload(RecruiterLead.sales)
                    .selectinload(Sales.clients)
                )
                .where(Recruiter.recruiter_id == recruiter_uuid)
            )
            recruiter_result = await db.execute(recruiter_stmt)
            recruiter = recruiter_result.scalar_one_or_none()

            if not recruiter:
                return json.dumps({
                    "error": f"Recruiter {recruiter_id} not found in database",
                    "success": False
                }, indent=2)

            # Determine which clients the recruiter has access to
            allowed_client_ids = set()
            if recruiter.recruiter_lead and recruiter.recruiter_lead.sales:
                # Recruiter has a sales org - only allow clients from that sales org
                sales = recruiter.recruiter_lead.sales
                allowed_client_ids = {client.client_id for client in sales.clients}
                logger.info(f"Recruiter {recruiter_id} has access to {len(allowed_client_ids)} clients from sales org {sales.sales_id}")
            else:
                # Recruiter has no sales org - allow all clients (fallback)
                logger.info(f"Recruiter {recruiter_id} has no RecruiterLead - allowing all clients")
                all_clients_stmt = select(Client.client_id)
                all_clients_result = await db.execute(all_clients_stmt)
                allowed_client_ids = {row[0] for row in all_clients_result.all()}

            # Check if the requested client is in the allowed list
            if client_uuid not in allowed_client_ids:
                return json.dumps({
                    "error": f"Access denied. Client {client_id} does not belong to your sales organization.",
                    "success": False
                }, indent=2)

            # Query database for Client with Company and Sales details
            stmt = (
                select(Client)
                .options(
                    selectinload(Client.company),
                    selectinload(Client.user),
                    selectinload(Client.sales).selectinload(Sales.user)
                )
                .where(Client.client_id == client_uuid)
            )
            result = await db.execute(stmt)
            client = result.scalar_one_or_none()

            if not client:
                return json.dumps({"error": f"Client {client_id} not found in database", "success": False}, indent=2)

            # Extract required data
            company_name = client.company.company_name if client.company else None
            client_email = client.user.email if client.user else None
            company_details = client.company.company_details if client.company else None

            # Validate required fields
            if not company_name:
                return json.dumps({"error": "Client has no company name", "success": False}, indent=2)
            if not client_email:
                return json.dumps({"error": "Client has no email address", "success": False}, indent=2)

            # Extract website from company_details JSONB (default to empty string)
            website = ""
            if company_details and isinstance(company_details, dict):
                website = company_details.get("website", "")

            # Determine first_name for client contact
            # Use client.user.full_name, fallback to company_name
            first_name = client.user.full_name if client.user and client.user.full_name else company_name

            logger.info(
                f"Creating CEIPAL client (two-step): '{company_name}', "
                f"email={client_email}, first_name='{first_name}', website='{website}'"
            )

            # Create CEIPAL client instance
            ceipal = CEIPAL(
                email=settings.ceipal_email,
                password=settings.ceipal_password,
                api_key=settings.ceipal_api_key,
                db=db,
            )

            # =====================================================================
            # STEP 1: Create Client
            # =====================================================================
            ceipal_client_id = None
            ceipal_client_contact_id = None

            try:
                logger.info("=== STEP 1: Creating client in CEIPAL ===")
                client_creation_result = await ceipal.create_client_in_ceipal(
                    client_name=company_name,
                    email_id=client_email,
                    website=website
                )
            except Exception as step1_error:
                logger.error(f"STEP 1 FAILED: {step1_error}")

                # Check if client already exists
                if "already exists" in str(step1_error).lower() or "duplicate" in str(step1_error).lower():
                    # Try to fetch existing IDs
                    from app.ceipal.schemas import CeipalAPIEndpoint
                    try:
                        lookup = await ceipal.lookup_email_in_api(
                            client_email,
                            CeipalAPIEndpoint.CLIENT_CONTACTS
                        )
                        if lookup.get("found") and lookup.get("id"):
                            # Client already exists, cache the existing contact ID
                            client.client_details = client.client_details or {}
                            client.client_details["ceipal_client_contact_id"] = lookup["id"]
                            client.client_details["ceipal_created_at"] = datetime.now(UTC).isoformat()
                            client.client_details["ceipal_verified_at"] = datetime.now(UTC).isoformat()
                            await db.commit()

                            return json.dumps({
                                "success": True,
                                "client_id": client_id,
                                "client_name": company_name,
                                "client_email": client_email,
                                "ceipal_client_id": None,  # Unknown (client already existed)
                                "ceipal_client_contact_id": lookup["id"],
                                "message": f"Client '{company_name}' already exists in CEIPAL. Contact ID cached successfully.",
                                "cached": True
                            }, indent=2)
                    except Exception as lookup_error:
                        logger.error(f"Failed to lookup existing client: {lookup_error}")

                return json.dumps({
                    "error": f"STEP 1 FAILED: {str(step1_error)}",
                    "success": False
                }, indent=2)

            # Check if Step 1 was successful
            if not client_creation_result.get("success"):
                error_message = client_creation_result.get("message", "Unknown error")
                logger.error(f"STEP 1 FAILED: {error_message}")
                return json.dumps({
                    "error": f"STEP 1 FAILED: {error_message}",
                    "success": False
                }, indent=2)

            ceipal_client_id = client_creation_result.get("client_id")

            if not ceipal_client_id:
                logger.error("STEP 1: Client creation succeeded but no client_id returned")
                return json.dumps({
                    "error": "STEP 1: Client creation succeeded but no client_id returned",
                    "success": False
                }, indent=2)

            logger.success(f"STEP 1 SUCCEEDED: Created client with ID {ceipal_client_id}")

            # =====================================================================
            # Look up job_owner (primary_owner) from client.sales.user.email
            # =====================================================================
            primary_owner_id = None
            if client.sales and client.sales.user and client.sales.user.email:
                try:
                    sales_email = client.sales.user.email
                    logger.info(f"Looking up job_owner from sales email: {sales_email}")
                    from app.ceipal.schemas import CeipalAPIEndpoint

                    # Look up sales user in CEIPAL using existing API
                    sales_lookup = await ceipal.lookup_email_in_api(
                        sales_email,
                        CeipalAPIEndpoint.SALES_MANAGER
                    )
                    if sales_lookup.get("found") and sales_lookup.get("id"):
                        primary_owner_id = sales_lookup["id"]
                        logger.success(f"Found job_owner CEIPAL ID: {primary_owner_id}")
                    else:
                        logger.warning(f"Sales user {sales_email} not found in CEIPAL. Will use default primary_owner.")
                except Exception as owner_lookup_error:
                    logger.warning(f"Failed to lookup job_owner: {owner_lookup_error}. Will use default primary_owner.")
            else:
                logger.info("No sales user assigned to client. Will use default primary_owner.")

            # =====================================================================
            # STEP 2: Create Client Contact (with retry on failure)
            # =====================================================================
            try:
                logger.info("=== STEP 2: Creating client contact in CEIPAL ===")
                contact_creation_result = await ceipal.create_client_contact_in_ceipal(
                    first_name=first_name,
                    email_id=client_email,
                    client_id=ceipal_client_id,
                    primary_owner_id=primary_owner_id,  # Pass job_owner or None (will use default)
                    max_retries=2  # Retry once (2 total attempts)
                )

                # Check if Step 2 was successful
                if not contact_creation_result.get("success"):
                    error_message = contact_creation_result.get("message", "Unknown error")
                    logger.error(f"STEP 2 FAILED after retries: {error_message}")

                    # "Forget" Step 1 - don't cache anything
                    logger.warning("Forgetting Step 1 due to Step 2 failure. Not caching any IDs.")

                    return json.dumps({
                        "error": (
                            f"STEP 2 FAILED: Could not create client contact after retries. "
                            f"Client was created (ID: {ceipal_client_id}) but contact creation failed. "
                            f"Error: {error_message}"
                        ),
                        "success": False,
                        "partial_success": True,
                        "ceipal_client_id": ceipal_client_id,
                        "step1_succeeded": True,
                        "step2_succeeded": False
                    }, indent=2)

                ceipal_client_contact_id = contact_creation_result.get("client_contact_id")

                if not ceipal_client_contact_id:
                    logger.error("STEP 2: Contact creation succeeded but no client_contact_id returned")
                    logger.warning("Forgetting Step 1 due to missing contact ID. Not caching any IDs.")

                    return json.dumps({
                        "error": "STEP 2: Contact creation succeeded but no client_contact_id returned",
                        "success": False,
                        "partial_success": True,
                        "ceipal_client_id": ceipal_client_id
                    }, indent=2)

                logger.success(f"STEP 2 SUCCEEDED: Created client contact with ID {ceipal_client_contact_id}")

            except Exception as step2_error:
                logger.error(f"STEP 2 EXCEPTION: {step2_error}")
                logger.warning("Forgetting Step 1 due to Step 2 exception. Not caching any IDs.")

                return json.dumps({
                    "error": (
                        f"STEP 2 FAILED: Exception during client contact creation. "
                        f"Client was created (ID: {ceipal_client_id}) but contact creation failed. "
                        f"Error: {str(step2_error)}"
                    ),
                    "success": False,
                    "partial_success": True,
                    "ceipal_client_id": ceipal_client_id,
                    "step1_succeeded": True,
                    "step2_succeeded": False
                }, indent=2)

            # =====================================================================
            # BOTH STEPS SUCCEEDED: Cache both IDs
            # =====================================================================
            try:
                client.client_details = client.client_details or {}
                client.client_details["ceipal_client_id"] = ceipal_client_id
                client.client_details["ceipal_client_contact_id"] = ceipal_client_contact_id
                client.client_details["ceipal_created_at"] = datetime.now(UTC).isoformat()
                client.client_details["ceipal_verified_at"] = datetime.now(UTC).isoformat()
                # Flag to indicate this client was just created (for job posting client_type)
                client.client_details["is_new_ceipal_client"] = True
                await db.commit()
                cached = True
                logger.success(
                    f"Successfully cached both IDs for client {client_id}: "
                    f"client_id={ceipal_client_id}, contact_id={ceipal_client_contact_id}"
                )
            except Exception as cache_error:
                logger.error(f"Failed to cache CEIPAL IDs: {cache_error}")
                cached = False
                # Still return success since both steps succeeded in CEIPAL

            return json.dumps({
                "success": True,
                "client_id": client_id,
                "client_name": company_name,
                "client_email": client_email,
                "ceipal_client_id": ceipal_client_id,
                "ceipal_client_contact_id": ceipal_client_contact_id,
                "message": (
                    f"Successfully created client '{company_name}' and contact in CEIPAL. "
                    f"Client ID: {ceipal_client_id}, Contact ID: {ceipal_client_contact_id}" +
                    ("" if cached else " (cache update failed, will sync on next verification)")
                ),
                "cached": cached
            }, indent=2)

    except Exception as e:
        logger.error(f"Error in create_ceipal_client: {e}")
        return json.dumps({"error": f"Error creating client: {str(e)}", "success": False}, indent=2)


@mcp.tool(
    description="Create a job requirement with client assignment and draft for recruiter. Requires client_id, recruiter_id, and all job details including min_pay_rate and publish_to (website to publish job). The currency will be automatically inferred from the country. IMPORTANT: Always ask the user which website they want to publish the job to - options are: 'TASC Global', 'TASC Saudi Arabia', 'Future Milez', or 'AIQU'. Creates JobRequirement, JobRequirementAssignment (auto-assigned to recruiter), and JobDraft linked together with country validation."
)
@ensure_json_response
async def create_job_requirement_with_draft(
    client_id: str,
    recruiter_id: str,
    input_data: JobRequestParseInput
) -> Union[GeneratedJD, str]:
    """
    Create a complete job requirement with assignment and draft in one step.

    This tool:
    1. Parses job details and generates JD using AI
    2. Creates JobRequirement linked to client
    3. Creates JobRequirementAssignment (recruiter auto-assigns to self)
    4. Creates JobDraft linked to the requirement
    5. Infers currency from country automatically

    Args:
        client_id: UUID string of the client company
        recruiter_id: UUID string of the recruiter
        input_data: Job details (title, location, skills, min_pay_rate, etc.)

    Returns:
        GeneratedJD with requirement_id and draft_id
    """
    logger.info(
        f"Tool 'create_job_requirement_with_draft' called for client: {client_id}, recruiter: {recruiter_id}"
    )

    try:
        client_uuid = uuid.UUID(client_id.strip())
        recruiter_uuid = uuid.UUID(recruiter_id.strip())
    except ValueError as exc:
        logger.error(f"Invalid UUID format: {exc}")
        raise HTTPException(status_code=400, detail="Invalid client_id or recruiter_id format") from exc

    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                # Step 1: Parse job request
                parsed_result = await JobParsingService.parse_job_request_detailed(input_data)
                logger.info(
                    f"Parsed job: {parsed_result.title} with {len(parsed_result.skills)} skills"
                )

                # Step 1.5: Infer pay_rate_currency from country if not provided
                # Store country validation result for reuse in Step 6
                country_validation_result = None
                pay_rate_currency = input_data.pay_rate_currency

                if not pay_rate_currency and parsed_result.location_country:
                    async with AsyncSessionLocal() as validation_session:
                        country_validation_result = await get_country_currency_from_ceipal(
                            parsed_result.location_country, validation_session
                        )

                    if not country_validation_result.get("found"):
                        # No exact match - return error with suggestions
                        suggestions = country_validation_result.get("suggestions", [])
                        if suggestions:
                            suggestion_text = "\n".join([
                                f"{i+1}. {s['name']} (Currency: {s.get('currency', 'N/A')})"
                                for i, s in enumerate(suggestions)
                            ])
                            error_msg = (
                                f"{country_validation_result.get('error', 'Country not found')}\n\n"
                                f"Please select one of these countries:\n{suggestion_text}\n\n"
                                f"Provide the full country name from the list above."
                            )
                        else:
                            error_msg = country_validation_result.get("error", f"Country '{parsed_result.location_country}' not found in Ceipal countries list.")

                        raise HTTPException(status_code=400, detail=error_msg)

                    pay_rate_currency = country_validation_result["country_id"]
                    logger.info(f"Inferred currency ID {pay_rate_currency} from country '{country_validation_result['country_name']}'")

                # Step 2: Generate JD using centralized service
                jd_input = GenerateJDInput(
                    title=parsed_result.title,
                    skills=parsed_result.skills,
                    location=parsed_result.location,
                    location_country=parsed_result.location_country,
                    experience=parsed_result.experience,
                    max_experience=parsed_result.max_experience,
                    domain=parsed_result.domain,
                    number_of_positions=parsed_result.number_of_positions,
                    time_to_fill_days=parsed_result.time_to_fill_days,
                    priority=parsed_result.priority,
                    pay_rate_currency=pay_rate_currency,
                    min_pay_rate=input_data.min_pay_rate,
                    job_requirement_id=None,  # Will be created
                )

                # Use centralized JD generation service (no draft creation)
                structured_jd, full_text_jd = await JobDescriptionService.generate_jd_for_requirement(
                    jd_input
                )

                # Step 3: Get recruiter's user_id and load recruiter_lead relationship
                from sqlalchemy.orm import selectinload

                recruiter_stmt = (
                    select(Recruiter)
                    .where(Recruiter.recruiter_id == recruiter_uuid)
                    .options(selectinload(Recruiter.recruiter_lead))
                )
                recruiter_result = await session.execute(recruiter_stmt)
                recruiter = recruiter_result.scalar_one_or_none()

                if not recruiter:
                    raise HTTPException(status_code=404, detail=f"Recruiter {recruiter_id} not found")

                # Look up recruiter's sales lead via relationship navigation
                sales_lead_id = None
                if recruiter.recruiter_lead and recruiter.recruiter_lead.sales_id:
                    sales_lead_id = recruiter.recruiter_lead.sales_id
                    logger.info(f"Found sales lead {sales_lead_id} for recruiter {recruiter_id}")
                else:
                    logger.warning(f"No sales lead found for recruiter {recruiter_id}, assignment will be marked as self-assigned")

                # Step 4: Create JobRequirement directly
                skills_for_storage = [skill.model_dump() for skill in jd_input.skills]

                # Create approval metadata for auto-approved recruiter-created job
                approval_metadata = {
                    "approve_at": datetime.now(timezone.utc).isoformat(),
                    "approve_by": str(recruiter.user_id),
                    "approve_notes": "Auto-approved: Recruiter direct job creation",
                    "approval_type": "AUTO_APPROVED_RECRUITER"
                }

                # Convert priority string to enum (REQUIRED)
                from app.db.schemas import JobPriority
                if not jd_input.priority:
                    raise ValueError("Priority is required and cannot be None")

                try:
                    priority_enum = JobPriority[jd_input.priority.upper()]
                    logger.info(f"✓ Priority validated: {jd_input.priority} → {priority_enum.value}")
                except (KeyError, AttributeError) as e:
                    raise ValueError(f"Invalid priority '{jd_input.priority}'. Must be HIGH, MEDIUM, or LOW") from e

                job_requirement = JobRequirement(
                    client_id=client_uuid,
                    created_by_user_id=recruiter.user_id,
                    created_by_role="RECRUITER",
                    title=jd_input.title,
                    location=jd_input.location,
                    domain=jd_input.domain,
                    openings=jd_input.number_of_positions or 1,
                    sla_days=jd_input.time_to_fill_days,
                    priority=priority_enum,
                    employment_type=input_data.employment_type,
                    experience=jd_input.experience,
                    max_experience=jd_input.max_experience,
                    skills=skills_for_storage,
                    structured_jd=structured_jd.model_dump(),
                    job_description_full_text=full_text_jd,
                    job_requirement_details=approval_metadata,
                    status="IN_PROGRESS",  # Auto-approved, ready for recruiter to work on
                )

                session.add(job_requirement)
                await session.flush()  # Flush to get the generated ID
                logger.info(f"Created JobRequirement: {job_requirement.job_requirement_id}")

                # Step 5: Create JobRequirementAssignment (auto-assign to recruiter)
                assignment_notes_text = "Auto-assigned during recruiter direct job creation."
                if sales_lead_id:
                    assignment_notes_text += " Sales lead tracked for audit purposes."
                else:
                    assignment_notes_text += " No sales lead found - marked as self-assigned."

                assignment = JobRequirementAssignment(
                    job_requirement_id=job_requirement.job_requirement_id,
                    recruiter_id=recruiter_uuid,
                    assigned_by_sales_id=sales_lead_id,  # Use sales lead ID if available, None if not
                    assignment_notes=assignment_notes_text
                )

                session.add(assignment)
                await session.flush()
                logger.info(f"Created JobRequirementAssignment for recruiter {recruiter_id}")

                # Step 6: Create pay_info using stored country validation result (reuse from Step 1.5)
                pay_info = None
                country_validation_message = None

                if jd_input.location_country:
                    # Reuse country validation from Step 1.5 if available (avoids duplicate CEIPAL call)
                    if country_validation_result and country_validation_result.get("found"):
                        # Use the already-validated country data
                        matched_country = country_validation_result["country_name"]
                        country_id = country_validation_result["country_id"]

                        country_validation_message = f"Country '{matched_country}' validated (ID: {country_id})"
                        logger.info(country_validation_message)

                        # Create pay_info
                        pay_info = {
                            "pay_rate_currency": jd_input.pay_rate_currency,
                            "min_pay_rate": jd_input.min_pay_rate,
                            "country": matched_country,
                            "country_id": country_id,
                            "user_input_country": jd_input.location_country,
                        }
                    else:
                        # Country provided but not validated (or validation failed in Step 1.5)
                        logger.warning(f"Country '{jd_input.location_country}' not validated")
                        country_validation_message = f"Warning: Country '{jd_input.location_country}' not found in Ceipal country list"

                        # Create pay_info without country validation
                        pay_info = {
                            "pay_rate_currency": jd_input.pay_rate_currency,
                            "min_pay_rate": jd_input.min_pay_rate,
                            "country": jd_input.location_country,
                            "country_id": None,
                            "user_input_country": jd_input.location_country,
                        }
                else:
                    # No location_country provided, just store pay rate info
                    pay_info = {
                        "pay_rate_currency": jd_input.pay_rate_currency,
                        "min_pay_rate": jd_input.min_pay_rate,
                    }
                    logger.warning("Pay rate provided but no location_country for validation")

                # Step 7: Map publish_to to website URL
                website_map = {
                    "TASC Global": "www.tascoutsourcing.com",
                    "TASC Saudi Arabia": "www.tascoutsourcing.sa",
                    "Future Milez": "www.futuremilez.com",
                    "AIQU": "www.aiqusearch.com",
                }
                publish_to_url = None
                if input_data.publish_to:
                    publish_to_url = website_map.get(input_data.publish_to, input_data.publish_to)
                    logger.info(f"Mapped publish_to '{input_data.publish_to}' to '{publish_to_url}'")

                # Add publish_to to structured_jd
                structured_jd_dict = structured_jd.model_dump()
                if publish_to_url:
                    structured_jd_dict['publish_to'] = publish_to_url

                # Convert priority string to enum explicitly for draft (REQUIRED)
                if not jd_input.priority:
                    raise ValueError("Priority is required and cannot be None for draft")

                try:
                    # Strip whitespace and convert to uppercase before enum lookup
                    priority_str = str(jd_input.priority).strip().upper()
                    draft_priority_enum = JobPriority[priority_str]
                    logger.info(f"✓ Draft priority converted: '{jd_input.priority}' → {draft_priority_enum.value}")
                except (KeyError, AttributeError) as e:
                    raise ValueError(f"Invalid priority '{jd_input.priority}' for draft. Must be HIGH, MEDIUM, or LOW") from e

                # Step 8: Create JobDraft linked to requirement
                draft = JobDraft(
                    title=jd_input.title,
                    structured_jd=structured_jd_dict,
                    full_text=full_text_jd,
                    skills=skills_for_storage,
                    location=jd_input.location,
                    location_country=jd_input.location_country,
                    experience=jd_input.experience,
                    max_experience=jd_input.max_experience,
                    domain=jd_input.domain,
                    number_of_positions=jd_input.number_of_positions or 1,
                    time_to_fill_days=jd_input.time_to_fill_days,
                    priority=draft_priority_enum,  # ✓ Explicit enum, not string
                    pay_info=pay_info,
                    job_requirement_id=job_requirement.job_requirement_id,
                )

                session.add(draft)
                await session.flush()  # Flush to get the generated draft_id
                draft_id = str(draft.draft_id)
                job_requirement_id_str = str(job_requirement.job_requirement_id)

                # Log country validation if present
                if country_validation_message:
                    logger.info(f"Country validation for draft {draft_id}: {country_validation_message}")

                logger.info(f"Created JobDraft: {draft_id}")

                logger.success(
                    f"Successfully created job requirement, assignment, and draft for {jd_input.title}"
                )

            # Transaction committed here - draft is now visible to other sessions

            # Step 7: Generate questions synchronously (after transaction commits)
            try:
                await JobDescriptionService._background_generate_questions(
                    draft_id, full_text_jd
                )
                logger.info(f"Successfully generated prescreening questions for draft {draft_id}")
            except Exception as e:
                logger.error(f"Failed to generate questions for draft {draft_id}: {e}")
                # Draft still exists, user can add questions manually later

            return GeneratedJD(
                title=jd_input.title,
                structured_jd=structured_jd,
                full_text=full_text_jd,
                process_id=uuid.UUID(draft_id),
                job_requirement_id=job_requirement_id_str,
            )

    except HTTPException as he:
        logger.error(f"HTTP error in create_job_requirement_with_draft: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error in create_job_requirement_with_draft: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create job requirement with draft: {e}"
        )


# @mcp.tool(
#     description="Populate job creation from an existing job requirement. Use format: job_requirement_id. This loads all data from the requirement (title, location, job_details, interview_process, etc.) and returns it for the agent to use in job creation."
# )
# async def populate_job_from_requirement(job_requirement_id: str) -> str:
#     """
#     Load all data from an existing job requirement to populate job creation fields.

#     This tool retrieves the full job requirement details including:
#     - Job title, location, employment type
#     - Job details (responsibilities, qualifications)
#     - Interview process/rounds
#     - Skills and experience requirements
#     - Priority level
#     - Client information

#     Args:
#         job_requirement_id: UUID string of the job requirement

#     Returns:
#         JSON string with all job requirement data formatted for job creation
#     """
#     logger.info(
#         f"Tool 'populate_job_from_requirement' called for requirement: {job_requirement_id}"
#     )

#     try:
#         async with AsyncSessionLocal() as session:
#             from app.db.models import JobRequirement
#             from sqlalchemy.orm import selectinload

#             # Query the job requirement with related data
#             query = (
#                 select(JobRequirement)
#                 .options(selectinload(JobRequirement.client))
#                 .where(
#                     JobRequirement.job_requirement_id == uuid.UUID(job_requirement_id)
#                 )
#             )

#             result = await session.execute(query)
#             job_req = result.scalar_one_or_none()

#             if not job_req:
#                 return json.dumps(
#                     {"error": f"Job requirement with ID {job_requirement_id} not found"}
#                 )

#             # Build the response with all available data
#             requirement_data = {
#                 "job_requirement_id": str(job_req.job_requirement_id),
#                 "title": job_req.title,
#                 "skills": job_req.skills,
#                 "min_exp": job_req.experience,
#                 "max_exp": job_req.max_experience,
#                 "structured_jd": job_req.structured_jd,
#                 "location": job_req.location,
#                 "employment_type": job_req.employment_type,
#                 "status": job_req.status.value
#                 if hasattr(job_req.status, "value")
#                 else str(job_req.status),
#                 "priority": job_req.priority.value
#                 if hasattr(job_req.priority, "value")
#                 else str(job_req.priority),
#                 "openings": job_req.openings,
#                 "sla_days": job_req.sla_days,
#                 "industry": job_req.industry,
#                 "client_info": {
#                     "client_id": str(job_req.client.client_id)
#                     if job_req.client
#                     else None,
#                 },
#                 "created_at": job_req.created_at.isoformat()
#                 if job_req.created_at
#                 else None,
#                 "message": "Job requirement data loaded successfully. Use this information to populate the job creation flow.",
#             }

#             return json.dumps(requirement_data, indent=2)

#     except ValueError as e:
#         logger.error(f"Invalid UUID in populate_job_from_requirement: {e}")
#         return json.dumps({"error": "Invalid job_requirement_id format"})
#     except Exception as e:
#         logger.error(f"Error in populate_job_from_requirement: {e}")
#         return json.dumps({"error": str(e)})


# =========================================================================
#                   CLIENT JOB CREATION TOOLS
# =========================================================================


@mcp.tool(
    description=(
        "Display a welcome card with action buttons for the client. "
        "Use this when the client greets you (hi, hello, hey, what can you do) or starts a new conversation. "
        "This shows a friendly welcome message with quick action buttons to guide them."
    )
)
async def show_client_welcome_card() -> str:
    """
    Display a welcome card component with action buttons for clients.
    Used for greetings and initial interactions.
    """
    logger.info("Tool 'show_client_welcome_card' called")
    try:
        return json.dumps(
            {
                "trigger": "show_client_welcome_card",
                "message": "Lets help you create job requirements quickly and easily!",
                "actions": [
                    {
                        "label": "Create Job Requirement",
                        "action": "create_job",
                        "description": "Post a new job opening",
                        "icon": "briefcase",
                    },
                    {
                        "label": "View My Jobs",
                        "action": "view_jobs",
                        "description": "See all your job postings",
                        "icon": "list",
                    },
                ],
            }
        )
    except Exception as e:
        logger.error(f"Error in show_client_welcome_card: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description=(
        "Trigger the job requirement form for the client to fill in all details. "
        "Use this when the client wants to create a new job requirement. "
        "This will display a form component in the UI to collect: title, "
        "location, industry, openings, SLA, priority, and employment type."
    )
)
async def trigger_client_job_requirement_form(client_id: str) -> str:
    """
    Trigger the Create Job Requirement form component for clients.
    Client creates job for themselves, so no client dropdown needed.
    """
    logger.info(
        f"Tool 'trigger_client_job_requirement_form' called for client: {client_id}"
    )

    try:
        # Validate UUID format
        try:
            client_uuid = uuid.UUID(client_id)
        except (ValueError, AttributeError) as e:
            logger.error(f"Invalid client_id format: {client_id} - {e}")
            return json.dumps(
                {
                    "error": f"Invalid client_id format: {client_id}. Must be a valid UUID."
                }
            )

        # Client creates job for themselves - no need to fetch client list
        return json.dumps(
            {
                "trigger": "show_client_job_requirement_form",
                "client_id": client_id,  # Auto-populated in form
                "message": "Please fill in the job requirement details in the form.",
            }
        )

    except Exception as e:
        logger.error(f"Error in trigger_client_job_requirement_form: {e}")
        return json.dumps({"error": str(e)})


# =========================================================================
#                   SALES GREETING TOOLS
# =========================================================================


@mcp.tool(
    description=(
        "Display a welcome card with action buttons for the sales person. "
        "Use this when the sales person greets you (hi, hello, hey, what can you do) or starts a new conversation. "
        "This shows a friendly welcome message with quick action buttons to guide them."
    )
)
async def show_sales_welcome_card() -> str:
    """
    Display a welcome card component with action buttons for sales users.
    Used for greetings and initial interactions.
    """
    logger.info("Tool 'show_sales_welcome_card' called")
    try:
        return json.dumps(
            {
                "trigger": "show_sales_welcome_card",
                "message": "Lets help you manage job requirements efficiently!",
                "actions": [
                    {
                        "label": "Create Job Requirement",
                        "action": "create_job",
                        "description": "Create a new job requirement for a client",
                        "icon": "briefcase",
                    },
                    {
                        "label": "View My Jobs",
                        "action": "view_cases",
                        "description": "See all your managed job requirements",
                        "icon": "list",
                    },
                ],
            }
        )
    except Exception as e:
        logger.error(f"Error in show_sales_welcome_card: {e}")
        return json.dumps({"error": str(e)})


# =========================================================================
#                   SALES JOB CREATION TOOLS
# =========================================================================


@mcp.tool(
    description=(
        "Trigger the job requirement form for the sales person to fill in all details. "
        "Use this when the sales person wants to create a new job requirement. "
        "This will display a form component in the UI to collect: client, title, "
        "location, industry, openings, SLA, priority, and employment type."
    )
)
async def trigger_job_requirement_form() -> str:
    """
    Trigger the Create Job Requirement form component.
    Frontend will fetch clients from GET /api/sales_chat/clients endpoint.
    """
    logger.info("Tool 'trigger_job_requirement_form' called")

    try:
        return json.dumps(
            {
                "trigger": "show_job_requirement_form",
                "message": "Please fill in the job requirement details in the form.",
            }
        )

    except Exception as e:
        logger.error(f"Error in trigger_job_requirement_form: {e}")
        return json.dumps({"error": str(e)})


# =========================================================================
#                   RECRUITER GREETING TOOLS
# =========================================================================


@mcp.tool(
    description=(
        "Display a welcome card with action buttons for the recruiter. "
        "Use this when the recruiter greets you (hi, hello, hey, what can you do) or starts a new conversation. "
        "This shows a friendly welcome message with quick action buttons to guide them."
    )
)
@ensure_json_response
async def show_recruiter_welcome_card() -> str:
    """
    Display a welcome card component with action buttons for recruiters.
    Used for greetings and initial interactions.
    """
    logger.info("Tool 'show_recruiter_welcome_card' called")
    try:
        return json.dumps({
            "trigger": "show_recruiter_welcome_card",
            "message": "Let's help you manage your hiring workflows efficiently!",
            "actions": [
                {
                    "label": "Create New Job",
                    "action": "create new job",
                    "description": "Start a new job listing and candidate screening",
                    "icon": "briefcase"
                },
                {
                    "label": "Work on Existing Jobs",
                    "action": "work on existing jobs",
                    "description": "Continue with your assigned job requirements",
                    "icon": "list"
                }
            ]
        })
    except Exception as e:
        logger.error(f"Error in show_recruiter_welcome_card: {e}")
        return json.dumps({"error": str(e)})


# =======================================================================
#                   SALES JOB CREATION TOOLS
# =======================================================================


@mcp.tool(
    description=(
        "Parse a sales job request and generate a job description draft. "
        "Sales user must provide: location, experience (min years), skills, "
        "number of positions, and time to fill (days). "
        "This creates the initial JD that the sales person will review."
    )
)
async def sales_create_job_from_text(
    input_data: JobRequestParseInput,
) -> Dict[str, Any]:
    """
    Step 1 for Sales: Parse job requirements and generate JD draft.
    This is the FIRST tool to call in the sales job creation flow.
    """
    logger.info("Tool 'sales_create_job_from_text' called - parsing and generating JD.")

    try:
        # Use the same parsing and generation logic as client
        parsed_result = await JobParsingService.parse_job_request_detailed(input_data)
        logger.info(
            f"Parsed job: {parsed_result.title} with "
            f"{len(parsed_result.skills)} detailed skills"
        )

        jd_input = GenerateJDInput(
            title=parsed_result.title,
            skills=parsed_result.skills,
            location=parsed_result.location,
            experience=parsed_result.experience,
            max_experience=parsed_result.max_experience,
            domain=parsed_result.domain,
            number_of_positions=parsed_result.number_of_positions,
            time_to_fill_days=parsed_result.time_to_fill_days,
            priority=parsed_result.priority,
            job_requirement_id=input_data.job_requirement_id,
        )

        logger.info(f"Generating JD with {len(jd_input.skills)} skills")

        async with AsyncSessionLocal() as session:
            async with session.begin():
                result = await JobDescriptionService.generate_jd(
                    db=session, input_data=jd_input
                )

                draft_id = str(result.process_id)
                logger.info(f"Generated draft with ID: {draft_id}")

                response_dict = {
                    "draft_id": draft_id,
                    "title": result.title,
                    "structured_jd": result.structured_jd.model_dump(),
                    "full_text": result.full_text,
                    "questions": [],
                    "skills": [skill.model_dump() for skill in jd_input.skills],
                    "location": jd_input.location,
                    "experience": jd_input.experience,
                    "max_experience": jd_input.max_experience,
                    "domain": jd_input.domain,
                    "number_of_positions": jd_input.number_of_positions,
                    "time_to_fill_days": jd_input.time_to_fill_days,
                    "expires_at": None,
                }

                logger.success(
                    f"Successfully created job description draft {draft_id} "
                    f"for sales with {len(jd_input.skills)} skills"
                )
                return response_dict

    except Exception as e:
        logger.error(f"Error in sales_create_job_from_text: {e}")
        logger.exception("Full traceback:")
        raise HTTPException(
            status_code=500, detail=f"Job description creation error: {str(e)}"
        )


@mcp.tool(
    description=(
        "Update an existing job requirement. Allows editing JD, skills, experience, and business fields. "
        "Format: job_requirement_id|updates_json"
        'Example: uuid|{\\"title\\": \\"Senior Developer\\", \\"skills\\": [{\\"skill_name\\": \\"Python\\", \\"experience_years\\": 5}]}'
    )
)
async def update_job_requirement(input_data: str) -> str:
    """
    PATCH update for job requirements.
    Format: job_requirement_id|updates_json
    """
    logger.info(f"Tool 'update_job_requirement' called with: {input_data}")

    try:
        parts = input_data.split("|", 1)
        if len(parts) < 2:
            return json.dumps(
                {"error": "Invalid format. Use: job_requirement_id|updates_json"}
            )

        job_req_id = uuid.UUID(parts[0].strip())
        updates_json = parts[1].strip()
        updates = json.loads(updates_json)

        # Convert skills if provided (from dicts to proper format)
        if "skills" in updates and updates["skills"]:
            # Ensure skills are in the right format
            normalized_skills = []
            for skill in updates["skills"]:
                normalized_skill = {
                    "skill_name": skill.get("skill_name") or skill.get("name"),
                    "skill_type": skill.get("skill_type")
                    or skill.get("type")
                    or "technical",
                    "experience_years": skill.get("experience_years"),
                }
                normalized_skills.append(normalized_skill)
            updates["skills"] = normalized_skills

        async with AsyncSessionLocal() as session:
            updated_job_req = await JobRequirementService.update_job_requirement(
                db=session, job_requirement_id=job_req_id, updates=updates
            )

            return json.dumps(
                {
                    "job_requirement_id": str(updated_job_req.job_requirement_id),
                    "message": "Job requirement updated successfully",
                    "status": "success",
                    "title": updated_job_req.title,
                },
                default=str,
            )

    except ValueError as e:
        logger.error(f"Validation error in update_job_requirement: {e}")
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error(f"Error in update_job_requirement: {e}")
        return json.dumps({"error": str(e)})


@mcp.tool(
    description=(
        "Parse a job request and create a job requirement directly with AI-generated JD (Sales flow). "
        "Format: raw_text|client_id|location|industry|openings|sla_days|priority|employment_type|sales_id|user_id"
    )
)
async def sales_create_job_requirement_direct(input_data: str) -> str:
    """
    NEW: Direct job requirement creation for sales (no draft).
    Format: raw_text|client_id|location|industry|openings|sla_days|priority|employment_type|sales_id|user_id
    """
    logger.info("Tool 'sales_create_job_requirement_direct' called - direct creation.")

    try:
        parts = input_data.split("|")
        if len(parts) < 10:
            return json.dumps(
                {
                    "error": "Invalid format. Use: raw_text|client_id|location|industry|openings|sla_days|priority|employment_type|sales_id|user_id"
                }
            )

        raw_text = parts[0].strip()
        client_id = uuid.UUID(parts[1].strip())
        location = parts[2].strip() if parts[2].strip() else None
        industry = parts[3].strip() if parts[3].strip() else None
        openings = int(parts[4].strip())
        sla_days = int(parts[5].strip()) if parts[5].strip() else None
        priority = parts[6].strip().upper()
        employment_type = parts[7].strip().upper()
        sales_id = uuid.UUID(parts[8].strip())  # Not used in creation, just for context
        user_id = uuid.UUID(parts[9].strip())

        # Parse job request
        parsed_result = await JobParsingService.parse_job_request_detailed(
            JobRequestParseInput(raw_text=raw_text)
        )
        logger.info(
            f"Parsed job: {parsed_result.title} with {len(parsed_result.skills)} skills"
        )

        # Generate JD directly (no draft)
        jd_input = GenerateJDInput(
            title=parsed_result.title,
            skills=parsed_result.skills,
            location=location or parsed_result.location,
            experience=parsed_result.experience,
            max_experience=parsed_result.max_experience,
            domain=parsed_result.domain,
        )

        (
            structured_jd,
            job_description_full_text,
        ) = await JobDescriptionService.generate_jd_for_requirement(jd_input)

        # Create job requirement directly (created_by_role = SALES)
        async with AsyncSessionLocal() as session:
            job_requirement = await JobRequirementService.create_with_ai_jd(
                db=session,
                client_id=client_id,
                user_id=user_id,
                created_by_role="SALES",  # Mark as created by sales
                title=parsed_result.title,
                location=location or parsed_result.location,
                industry=industry,
                openings=openings,
                sla_days=sla_days,
                priority=priority,
                employment_type=employment_type,
                experience=parsed_result.experience,
                max_experience=parsed_result.max_experience,
                skills=[skill.model_dump() for skill in parsed_result.skills],
                structured_jd=structured_jd.model_dump(),
                job_description_full_text=job_description_full_text,
            )

            logger.success(
                f"Created job requirement {job_requirement.job_requirement_id} via sales"
            )

            return json.dumps(
                {
                    "job_requirement_id": str(job_requirement.job_requirement_id),
                    "title": job_requirement.title,
                    "message": "Job requirement created successfully via sales!",
                    "status": "success",
                    "created_by_role": "SALES",
                },
                default=str,
            )

    except Exception as e:
        logger.error(f"Error in sales_create_job_requirement_direct: {e}")
        return json.dumps({"error": str(e)})


# DEPRECATED TOOL - Use sales_create_job_requirement_direct instead
@mcp.tool(
    description=(
        "DEPRECATED: Creates a job requirement after the JD has been approved (Sales flow). "
        "Use sales_create_job_requirement_direct instead. "
        "Format: draft_id|client_id|title|location|industry|openings|sla_days|priority|employment_type|sales_id|user_id"
    )
)
async def sales_create_job_requirement_from_jd_draft(input_data: str) -> str:
    """
    Step 2 for Sales: Create job requirement with all business details.
    Format: draft_id|client_id|title|location|industry|openings|sla_days|priority|employment_type|sales_id|user_id

    The sales_id and user_id are required to track who created the job requirement.
    """
    logger.info(
        f"Tool 'sales_create_job_requirement_from_jd_draft' called with: {input_data}"
    )

    try:
        parts = input_data.split("|")
        if len(parts) < 11:
            return json.dumps(
                {
                    "error": "Invalid format. Use: draft_id|client_id|title|location|industry|openings|sla_days|priority|employment_type|sales_id|user_id"
                }
            )

        draft_id = parts[0].strip()
        client_id = uuid.UUID(parts[1].strip())
        title = parts[2].strip()
        location = parts[3].strip() if parts[3].strip() else None
        industry = parts[4].strip() if parts[4].strip() else None
        openings = int(parts[5].strip())
        sla_days = int(parts[6].strip()) if parts[6].strip() else None
        priority = parts[7].strip().upper()
        employment_type = parts[8].strip().upper()
        sales_id = uuid.UUID(parts[9].strip())
        user_id = uuid.UUID(parts[10].strip())

        async with AsyncSessionLocal() as session:
            # Get the job draft
            stmt = select(JobDraft).where(JobDraft.draft_id == uuid.UUID(draft_id))
            result = await session.execute(stmt)
            job_draft = result.scalar_one_or_none()

            if not job_draft:
                return json.dumps({"error": f"Draft with ID {draft_id} not found."})

            jd_details = job_draft.structured_jd

            # Convert priority string to enum (REQUIRED)
            from app.db.schemas import JobPriority
            if not priority:
                raise ValueError("Priority is required and cannot be None")

            try:
                priority_enum = JobPriority[priority.upper()]
                logger.info(f"✓ Priority validated: {priority} → {priority_enum.value}")
            except (KeyError, AttributeError) as e:
                raise ValueError(f"Invalid priority '{priority}'. Must be HIGH, MEDIUM, or LOW") from e

            # Create job requirement with sales tracking
            new_job_requirement = JobRequirement(
                client_id=client_id,
                title=title,
                location=location,
                industry=industry,
                openings=openings,
                sla_days=sla_days,
                priority=priority_enum,
                employment_type=employment_type,
                status=JobRequirementStatus.OPEN,
                job_requirement_details=jd_details,
                created_by_user_id=user_id,  # Track who created it
                created_by_role="SALES",  # Mark as created by sales
            )

            session.add(new_job_requirement)
            await session.commit()
            await session.refresh(new_job_requirement)

            logger.success(
                f"Successfully created job requirement "
                f"{new_job_requirement.job_requirement_id} via sales flow by user {user_id}"
            )

            return json.dumps(
                {
                    "job_requirement_id": str(new_job_requirement.job_requirement_id),
                    "message": "Job requirement created successfully. Would you like to assign a recruiter to this job?",
                    "status": "success",
                    "created_by_role": "SALES",
                },
                default=str,
            )

    except Exception as e:
        logger.error(f"Error creating job requirement: {e}")
        return json.dumps({"error": str(e)})


# =======================================================================
#                   SALES MANAGEMENT TOOLS
# =======================================================================


@mcp.tool(
    description=(
        "Trigger to display all job requirements (cases) for this sales person. "
        "This will show the job requirements list component in the UI with all details "
        "including client info, status, priority, and assigned recruiters. "
        "The frontend will fetch the actual data from the database."
    )
)
async def get_sales_job_requirements() -> str:
    """
    Lightweight trigger to display job requirements list in the UI.
    The frontend will handle fetching the actual data via GET API.
    """
    logger.info("Tool 'get_sales_job_requirements' called")

    try:
        # Just return a trigger - frontend will fetch the data via authenticated API
        return json.dumps(
            {
                "success": True,
                "message": "Fetching job requirements for sales person. The cases will be displayed in the UI.",
                "trigger": "show_job_requirements_list",
            }
        )

    except Exception as e:
        logger.error(f"Error in get_sales_job_requirements: {e}")
        return json.dumps({"success": False, "error": str(e)})


# DEPRECATED: This tool is no longer used. Job approval is now handled via frontend API.
# Use POST /sales-dashboard/job-requirements/action with action="approve" instead.
# Keeping this commented out for reference.
#
# @mcp.tool(
#     description=(
#         "DEPRECATED: Use frontend API instead. "
#         "POST /sales-dashboard/job-requirements/action"
#     )
# )
# async def approve_job_requirement(input_data: str) -> str:
#     """DEPRECATED: Approve via POST /sales-dashboard/job-requirements/action"""
#     logger.warning("approve_job_requirement tool is deprecated. Use frontend API.")
#     return json.dumps({
#         "error": "This tool is deprecated. Please use the frontend UI to approve jobs.",
#         "api_endpoint": "POST /sales-dashboard/job-requirements/action",
#         "action": "approve"
#     })


# DEPRECATED: This tool is no longer used. Job on-hold is now handled via frontend API.
# Use POST /sales-dashboard/job-requirements/action with action="on_hold" instead.
# Keeping this commented out for reference.
#
# @mcp.tool(
#     description=(
#         "DEPRECATED: Use frontend API instead. "
#         "POST /sales-dashboard/job-requirements/action"
#     )
# )
# async def put_job_on_hold(input_data: str) -> str:
#     """DEPRECATED: Put on hold via POST /sales-dashboard/job-requirements/action"""
#     logger.warning("put_job_on_hold tool is deprecated. Use frontend API.")
#     return json.dumps({
#         "error": "This tool is deprecated. Please use the frontend UI to put jobs on hold.",
#         "api_endpoint": "POST /sales-dashboard/job-requirements/action",
#         "action": "on_hold"
#     })


@mcp.tool(
    description=(
        "Get all recruiters available for assignment (associated with this sales person "
        "through recruiter leads)."
    )
)
async def get_available_recruiters_for_sales() -> str:
    """Get available recruiters for this sales person.
    Frontend will fetch recruiters from GET /api/sales_chat/recruiters endpoint.
    """
    logger.info("Tool 'get_available_recruiters_for_sales' called")

    try:
        # Just return a trigger - frontend will fetch the data via authenticated API
        return json.dumps(
            {
                "success": True,
                "trigger": "show_recruiters_list",
                "message": "Fetching available recruiters for assignment.",
            }
        )

    except Exception as e:
        logger.error(f"Error in get_available_recruiters_for_sales: {e}")
        return json.dumps({"success": False, "error": str(e)})


# DEPRECATED: This tool is no longer used. Recruiter assignment is now handled via frontend API.
# Use POST /sales-dashboard/job-requirements/assign instead.
# Keeping this commented out for reference.
#
# @mcp.tool(
#     description=(
#         "DEPRECATED: Use frontend API instead. "
#         "POST /sales-dashboard/job-requirements/assign"
#     )
# )
# async def assign_recruiter_to_job(input_data: str) -> str:
#     """DEPRECATED: Assign via POST /sales-dashboard/job-requirements/assign"""
#     logger.warning("assign_recruiter_to_job tool is deprecated. Use frontend API.")
#     return json.dumps({
#         "error": "This tool is deprecated. Please use the frontend UI to assign recruiters.",
#         "api_endpoint": "POST /sales-dashboard/job-requirements/assign"
#     })
