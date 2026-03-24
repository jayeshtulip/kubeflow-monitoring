"""Jira API tool for the Executor agent."""
from __future__ import annotations
import os
import httpx
from pipelines.components.shared.base import get_logger

logger = get_logger(__name__)


def search_jira_tickets(
    jql_query: str,
    max_results: int = 10,
) -> list[dict]:
    """
    Search Jira using JQL.
    Returns list of ticket summaries with key, summary, status, description.
    """
    jira_url   = os.environ.get("JIRA_URL", "")
    jira_token = os.environ.get("JIRA_TOKEN", "")
    if not jira_url or not jira_token:
        return [{"error": "JIRA_URL or JIRA_TOKEN not configured", "tool": "jira"}]
    try:
        r = httpx.get(
            f"{jira_url}/rest/api/3/search",
            params={"jql": jql_query, "maxResults": max_results,
                    "fields": "summary,status,description,priority,assignee,created,updated"},
            headers={"Authorization": f"Bearer {jira_token}",
                     "Accept": "application/json"},
            timeout=15.0,
        )
        r.raise_for_status()
        issues = r.json().get("issues", [])
        results = []
        for issue in issues:
            fields = issue.get("fields", {})
            desc = fields.get("description", {})
            desc_text = ""
            if isinstance(desc, dict):
                for block in desc.get("content", []):
                    for item in block.get("content", []):
                        desc_text += item.get("text", "") + " "
            results.append({
                "key":         issue.get("key"),
                "summary":     fields.get("summary", ""),
                "status":      fields.get("status", {}).get("name", ""),
                "priority":    fields.get("priority", {}).get("name", ""),
                "description": desc_text[:500],
                "created":     fields.get("created", ""),
                "tool":        "jira",
            })
        logger.info("jira: %d tickets for jql=%s", len(results), jql_query[:60])
        return results
    except Exception as exc:
        logger.error("Jira error: %s", exc)
        return [{"error": str(exc), "tool": "jira"}]


def get_jira_ticket(ticket_key: str) -> dict:
    """Get a single Jira ticket by key (e.g. OPS-1234)."""
    results = search_jira_tickets(f"key = {ticket_key}", max_results=1)
    return results[0] if results else {"error": f"Ticket {ticket_key} not found"}