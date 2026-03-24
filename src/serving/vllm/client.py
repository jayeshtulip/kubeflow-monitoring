"""
vLLM HTTP client — Llama 2 7B GPTQ on H100.
OpenAI-compatible /v1/chat/completions endpoint.
Used by: high-throughput inference, P11 evaluation.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
import httpx
from pipelines.components.shared.base import EnvConfig, get_logger

logger = get_logger(__name__)


@dataclass
class VLLMResponse:
    text: str
    model: str
    latency_ms: float
    tokens_generated: int
    success: bool
    error: str = ""

    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms <= 0:
            return 0.0
        return round(self.tokens_generated / (self.latency_ms / 1000), 1)


class VLLMClient:
    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        self.base_url = self.cfg.vllm_api_base
        self.default_model = "meta-llama/Llama-2-7b-chat-hf"

    def chat(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> VLLMResponse:
        """Call vLLM /v1/chat/completions (OpenAI-compatible)."""
        model = model or self.default_model
        t0 = time.perf_counter()
        payload = {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      False,
        }
        try:
            r = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            latency = round((time.perf_counter() - t0) * 1000, 2)
            logger.info("vLLM %s: %.0fms %d tok (%.0f tok/s)",
                        model, latency, tokens,
                        tokens / (latency / 1000) if latency > 0 else 0)
            return VLLMResponse(text=text, model=model, latency_ms=latency,
                                tokens_generated=tokens, success=True)
        except Exception as exc:
            latency = round((time.perf_counter() - t0) * 1000, 2)
            logger.error("vLLM error: %s", exc)
            return VLLMResponse(text="", model=model, latency_ms=latency,
                                tokens_generated=0, success=False, error=str(exc))

    def health_check(self, timeout: float = 5.0) -> bool:
        """Returns True if vLLM /health endpoint responds 200."""
        try:
            r = httpx.get(f"{self.base_url.rstrip('/v1')}/health", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False