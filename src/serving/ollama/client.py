"""
Ollama HTTP client — Mistral-7B in-cluster inference.
Used by: Planner, Executor, Critic agents, P09 answer generation.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
import httpx
from pipelines.components.shared.base import EnvConfig, get_logger

logger = get_logger(__name__)


@dataclass
class OllamaResponse:
    text: str
    model: str
    latency_ms: float
    success: bool
    error: str = ""


class OllamaClient:
    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        self.base_url = self.cfg.ollama_base_url
        self.default_model = "mistral:7b"

    def generate(
        self,
        prompt: str,
        system: str = "",
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 90.0,
    ) -> OllamaResponse:
        model = model or self.default_model
        t0 = time.perf_counter()
        payload = {
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        try:
            r = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )
            r.raise_for_status()
            text = r.json().get("response", "").strip()
            latency = round((time.perf_counter() - t0) * 1000, 2)
            logger.info("Ollama %s: %.0fms, %d chars", model, latency, len(text))
            return OllamaResponse(text=text, model=model,
                                  latency_ms=latency, success=True)
        except Exception as exc:
            latency = round((time.perf_counter() - t0) * 1000, 2)
            logger.error("Ollama error: %s", exc)
            return OllamaResponse(text="", model=model,
                                  latency_ms=latency, success=False, error=str(exc))

    def chat(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 90.0,
    ) -> OllamaResponse:
        """OpenAI-compatible chat format — converts to Ollama prompt."""
        prompt_parts = []
        system = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt_parts.append("Assistant:")
        return self.generate(
            prompt="\n".join(prompt_parts),
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )