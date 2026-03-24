"""Pydantic schemas for the query API."""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    workflow: Optional[str] = Field(None, description="Override workflow: simple/react/planner_executor_critic")
    session_id: Optional[str] = Field(None)
    top_k: int = Field(5, ge=1, le=20)


class QueryResponse(BaseModel):
    response:         str
    workflow_used:    str
    latency_ms:       float
    session_id:       Optional[str] = None
    complexity_score: Optional[int] = None
    tool_calls:       int = 0
    replan_count:     int = 0
    success:          bool = True
    error:            Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str = "2.0.0"
    ollama_available:  bool = False
    vllm_available:    bool = False
    qdrant_available:  bool = False