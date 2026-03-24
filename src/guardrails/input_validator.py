"""
Input validation and guardrails.
Checks: prompt injection, PII, malicious intent, query classification.
Used by: FastAPI middleware before any LLM call.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from pipelines.components.shared.base import get_logger

logger = get_logger(__name__)


class RiskLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    BLOCKED = "blocked"


@dataclass
class ValidationResult:
    passed: bool
    risk_level: RiskLevel
    reasons: list[str] = field(default_factory=list)
    sanitised_query: str = ""


# Prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore (previous|all|prior) (instructions?|context|rules?)",
    r"(system|admin|root)\s*prompt",
    r"(forget|disregard|override)\s+(all|previous|your)",
    r"you are now\s+\w+",
    r"</?(s|inst|sys|system)>",
    r"new\s+instructions?\s*:",
    r"jailbreak",
    r"dan\s+mode",
    r"act as if you have no",
    r"pretend you (are|have) no (rules?|restrictions?|guidelines?)",
]

# PII patterns
PII_PATTERNS = {
    "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[- ]){3}\d{4}\b",
    "password_kv": r"(?i)(password|passwd|pwd|secret)\s*[=:]\s*\S+",
    "aws_key":     r"(?i)(aws[_-]?(secret|access)[_-]?key)\s*[=:]\s*\S+",
    "bearer_token": r"Bearer\s+[A-Za-z0-9\-_\.]{20,}",
    "private_key": r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
}

# Malicious intent keywords
MALICIOUS_KEYWORDS = [
    "drop table", "delete from", "truncate table",
    "rm -rf", "sudo rm", "chmod 777",
    "exec(", "eval(", "__import__",
    "os.system", "subprocess.call",
    "/etc/passwd", "/etc/shadow",
    "base64.decode", "pickle.loads",
]


def validate_input(query: str) -> ValidationResult:
    """
    Validate and classify an incoming query.
    Returns ValidationResult with pass/fail, risk level, and reasons.
    """
    if not query or not query.strip():
        return ValidationResult(
            passed=False,
            risk_level=RiskLevel.BLOCKED,
            reasons=["Empty query"],
        )

    reasons: list[str] = []
    risk = RiskLevel.LOW

    q_lower = query.lower()

    # Check prompt injection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            reasons.append(f"Prompt injection detected: {pattern}")
            risk = RiskLevel.BLOCKED

    # Check PII
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, query):
            reasons.append(f"PII detected: {pii_type}")
            risk = RiskLevel.BLOCKED

    # Check malicious intent
    for keyword in MALICIOUS_KEYWORDS:
        if keyword.lower() in q_lower:
            reasons.append(f"Malicious keyword: {keyword}")
            if risk != RiskLevel.BLOCKED:
                risk = RiskLevel.HIGH

    # Length check
    if len(query) > 4000:
        reasons.append(f"Query too long: {len(query)} chars (max 4000)")
        if risk == RiskLevel.LOW:
            risk = RiskLevel.MEDIUM

    passed = risk not in (RiskLevel.BLOCKED, RiskLevel.HIGH)

    if not reasons:
        risk = classify_risk(query)

    sanitised = query.strip()[:4000] if passed else ""

    if not passed:
        logger.warning("Input blocked: %s | reasons: %s", query[:80], reasons)
    else:
        logger.debug("Input accepted: risk=%s query=%s", risk.value, query[:60])

    return ValidationResult(
        passed=passed,
        risk_level=risk,
        reasons=reasons,
        sanitised_query=sanitised,
    )


def classify_risk(query: str) -> RiskLevel:
    """Classify query complexity/risk for routing purposes."""
    q = query.lower()
    complex_signals = [
        "why", "root cause", "intermittent", "timeout", "crash",
        "failing", "down", "outage", "investigate", "debug",
    ]
    medium_signals = [
        "how do i", "what is", "configure", "setup", "explain",
    ]
    if any(s in q for s in complex_signals):
        return RiskLevel.HIGH  # HIGH = complex routing (not a risk)
    if any(s in q for s in medium_signals):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW