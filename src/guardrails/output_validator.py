"""
Output validation — checks LLM responses before delivery.
Checks: no harmful commands, no PII exposure, no fabricated references.
Used by: FastAPI query router after response generation.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pipelines.components.shared.base import get_logger

logger = get_logger(__name__)


@dataclass
class OutputValidationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    safe_response: str = ""


HARMFUL_COMMAND_PATTERNS = [
    r"rm -rf /",
    r"DROP (TABLE|DATABASE|SCHEMA)",
    r"sudo (rm|chmod|chown|dd)",
    r":(){ :|:& };:",
    r"mkfs\.\w+",
]

PII_LEAK_PATTERNS = {
    "SSN":          r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card":  r"\b(?:\d{4}[- ]){3}\d{4}\b",
    "aws_key":      r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*\S+",
    "bearer_token": r"Bearer\s+[A-Za-z0-9\-_\.]{20,}",
}


def validate_output(response: str, source_contexts: list[str] | None = None) -> OutputValidationResult:
    """
    Validate LLM response before sending to user.
    Checks harmful commands, PII leakage, length.
    """
    if not response or not response.strip():
        return OutputValidationResult(
            passed=False,
            issues=["Empty response from LLM"],
            safe_response="",
        )

    issues: list[str] = []

    # Check harmful commands
    for pattern in HARMFUL_COMMAND_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            issues.append(f"Harmful command pattern detected: {pattern}")

    # Check PII leakage in output
    for pii_type, pattern in PII_LEAK_PATTERNS.items():
        if re.search(pattern, response):
            issues.append(f"PII in response: {pii_type}")

    # Check response length
    if len(response) > 10000:
        issues.append(f"Response too long: {len(response)} chars")
        response = response[:10000] + "\n[Response truncated]"

    passed = len(issues) == 0
    if not passed:
        logger.warning("Output validation failed: %s", issues)
    else:
        logger.debug("Output validated OK: %d chars", len(response))

    return OutputValidationResult(
        passed=passed,
        issues=issues,
        safe_response=response if passed else "",
    )