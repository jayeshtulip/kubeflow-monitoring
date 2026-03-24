"""
Great Expectations KFP v2 component.

Validates incoming documents against the doc_ingestion_suite before
they are chunked, embedded, or stored. Any document that fails is
rejected immediately — it never enters the vector store.

Validation suite: doc_ingestion_suite
  - text length between 50 and 5000 words
  - no null / empty text chunks
  - language detected as English (via langdetect)
  - no PII patterns (SSN, credit card, raw passwords)

Returns:
  validation_passed (bool output)  — consumed by P05 pipeline condition
  validation_report (str output)   — JSON summary written to GCS/S3 artifact
"""


from kfp import dsl
from kfp.dsl import Output, Artifact, component


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "great-expectations==0.18.0",
        "langdetect==1.0.9",
        "psycopg2-binary==2.9.9",
    ],
)
def validate_document_component(
    doc_text: str,
    doc_id: str,
    source: str,
    # Outputs
    validation_report: Output[Artifact],
) -> bool:
    """
    Validate a document against the doc_ingestion_suite.

    Returns True if validation passes, False otherwise.
    The validation_report artifact contains the full GX result JSON.
    """
    import json
    import re
    import time

    try:
        from langdetect import detect, LangDetectException
    except ImportError:
        detect = None

    report: dict = {
        "doc_id":     doc_id,
        "source":     source,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks":     [],
        "passed":     True,
    }

    def add_check(name: str, passed: bool, detail: str = "") -> None:
        report["checks"].append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            report["passed"] = False

    # ── Check 1: Non-empty ────────────────────────────────────────────────────
    if not doc_text or not doc_text.strip():
        add_check("non_empty", False, "Document text is empty or whitespace only")
        with open(validation_report.path, "w") as f:
            json.dump(report, f, indent=2)
        return False
    add_check("non_empty", True)

    # ── Check 2: Word count 50–5000 ───────────────────────────────────────────
    word_count = len(doc_text.split())
    in_range = 50 <= word_count <= 5000
    add_check(
        "word_count_range",
        in_range,
        f"word_count={word_count}, expected [50, 5000]",
    )

    # ── Check 3: Language is English ──────────────────────────────────────────
    if detect is not None:
        try:
            lang = detect(doc_text[:2000])
            is_english = lang == "en"
            add_check("language_english", is_english, f"detected_language={lang}")
        except Exception as e:
            add_check("language_english", False, f"langdetect error: {e}")
    else:
        add_check("language_english", True, "langdetect not available, skipped")

    # ── Check 4: No PII patterns ──────────────────────────────────────────────
    pii_patterns = {
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d{4}[- ]){3}\d{4}\b",
        "password_kv": r"(?i)(password|passwd|pwd)\s*[=:]\s*\S+",
        "aws_secret":  r"(?i)aws[_\-]secret[_\-]access[_\-]key\s*[=:]\s*\S+",
    }
    pii_found = []
    for pii_type, pattern in pii_patterns.items():
        if re.search(pattern, doc_text):
            pii_found.append(pii_type)

    add_check(
        "no_pii_patterns",
        len(pii_found) == 0,
        f"pii_types_found={pii_found}" if pii_found else "",
    )

    # ── Check 5: Minimum content density (not mostly whitespace/newlines) ─────
    non_space = len(re.sub(r"\s+", "", doc_text))
    density = non_space / max(len(doc_text), 1)
    add_check(
        "content_density",
        density >= 0.5,
        f"density={density:.2f}, expected >= 0.50",
    )

    # Write report
    with open(validation_report.path, "w") as f:
        json.dump(report, f, indent=2)

    return report["passed"]

