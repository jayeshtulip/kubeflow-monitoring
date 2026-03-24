"""
FastAPI application entry point.
Registers routers, middleware, startup/shutdown hooks.
"""
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routers import query, health
from pipelines.components.shared.base import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="LLM Platform API",
    description="Enterprise incident resolution via multi-agent LangGraph workflows.",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, tags=["health"])
app.include_router(query.router, prefix="/api", tags=["query"])


@app.on_event("startup")
async def startup():
    logger.info("LLM Platform API v2.0.0 starting up")


@app.on_event("shutdown")
async def shutdown():
    logger.info("LLM Platform API shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)