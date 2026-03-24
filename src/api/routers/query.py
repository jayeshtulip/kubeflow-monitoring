"""Main /api/query endpoint — routes to the correct LangGraph workflow."""
from __future__ import annotations
import time
import uuid
from fastapi import APIRouter, HTTPException, Request
from src.api.schemas.query import QueryRequest, QueryResponse
from src.agents.router.workflow_router import route_query, WorkflowType
from src.guardrails.input_validator import validate_input, RiskLevel
from src.guardrails.output_validator import validate_output
from pipelines.components.shared.base import EnvConfig, get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, http_request: Request) -> QueryResponse:
    """
    Main query endpoint.
    1. Validate input (guardrails)
    2. Route to workflow
    3. Execute workflow
    4. Validate output
    5. Return response
    """
    t_start = time.perf_counter()
    session_id = request.session_id or str(uuid.uuid4())[:8]
    cfg = EnvConfig()

    # Step 1: Input validation
    validation = validate_input(request.query)
    if not validation.passed:
        logger.warning("Query blocked session=%s reasons=%s", session_id, validation.reasons)
        raise HTTPException(
            status_code=400,
            detail={"blocked": True, "reasons": validation.reasons},
        )

    query = validation.sanitised_query

    # Step 2: Route to workflow
    if request.workflow:
        workflow_override = request.workflow
    else:
        routing = route_query(query)
        workflow_override = routing.workflow.value

    # Step 3: Execute workflow
    try:
        if workflow_override == WorkflowType.PLANNER_EXECUTOR_CRITIC.value:
            from src.workflows.planner_executor_critic.workflow import run as run_pec
            result = run_pec(query, cfg=cfg)
            response_text  = result.response
            tool_calls     = result.tool_call_count
            replan_count   = result.replan_count
            latency_ms     = result.total_latency_ms

        else:
            from src.workflows.simple_research.workflow import run as run_simple
            result = run_simple(query, top_k=request.top_k, cfg=cfg)
            response_text = result.response
            tool_calls    = 0
            replan_count  = 0
            latency_ms    = result.latency_ms

    except Exception as exc:
        logger.exception("Workflow error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Step 4: Output validation
    out_validation = validate_output(response_text)
    if not out_validation.passed:
        logger.warning("Output blocked session=%s issues=%s",
                       session_id, out_validation.issues)
        response_text = "I was unable to generate a safe response. Please rephrase your query."

    total_latency = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info("Query complete session=%s workflow=%s latency=%.0fms",
                session_id, workflow_override, total_latency)

    return QueryResponse(
        response=response_text,
        workflow_used=workflow_override,
        latency_ms=total_latency,
        session_id=session_id,
        tool_calls=tool_calls,
        replan_count=replan_count,
        success=True,
    )