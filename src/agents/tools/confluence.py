"""Confluence API tool for the Executor agent."""
from __future__ import annotations
import os
import httpx
from pipelines.components.shared.base import get_logger

logger = get_logger(__name__)


def search_confluence(
    query: str,
    space_key: str = "",
    max_results: int = 5,
) -> list[dict]:
    """
    Search Confluence pages using CQL.
    Returns list of pages with title, url, excerpt.
    """
    conf_url   = os.environ.get("CONFLUENCE_URL", "")
    conf_token = os.environ.get("CONFLUENCE_TOKEN", "")
    if not conf_url or not conf_token:
        return [{"error": "CONFLUENCE_URL or CONFLUENCE_TOKEN not configured",
                 "tool": "confluence"}]
    cql = f'text ~ "{query}"'
    if space_key:
        cql += f' AND space = "{space_key}"'
    cql += " ORDER BY lastmodified DESC"
    try:
        r = httpx.get(
            f"{conf_url}/rest/api/content/search",
            params={"cql": cql, "limit": max_results,
                    "expand": "body.storage,version,space"},
            headers={"Authorization": f"Bearer {conf_token}",
                     "Accept": "application/json"},
            timeout=15.0,
        )
        r.raise_for_status()
        results = []
        for page in r.json().get("results", []):
            body = page.get("body", {}).get("storage", {}).get("value", "")
            import re
            text = re.sub(r"<[^>]+>", " ", body)[:500]
            results.append({
                "title":   page.get("title", ""),
                "url":     f"{conf_url}/wiki{page.get('_links', {}).get('webui', '')}",
                "excerpt": text.strip(),
                "space":   page.get("space", {}).get("key", ""),
                "version": page.get("version", {}).get("number", 0),
                "tool":    "confluence",
            })
        logger.info("confluence: %d pages for query=%s", len(results), query[:60])
        return results
    except Exception as exc:
        logger.error("Confluence error: %s", exc)
        return [{"error": str(exc), "tool": "confluence"}]