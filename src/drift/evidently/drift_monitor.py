"""
Evidently AI drift monitor.
Detects:
  - Data drift: JS-divergence of query embedding distributions
  - Concept drift: RAGAS faithfulness degradation over 7-day rolling window

Triggered every 6 hours by Kubeflow P04 (Quality Monitoring).
Alerts via Prometheus counter + Slack webhook when thresholds exceeded.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from pipelines.components.shared.base import EnvConfig, get_logger, get_mlflow_client

logger = get_logger(__name__)

JS_DIV_THRESHOLD       = 0.15   # data drift alert
CONCEPT_DRIFT_DROP_PCT = 0.10   # 10% relative drop in faithfulness


@dataclass
class DriftReport:
    data_drift_score:    float
    concept_drift_score: float
    data_drift_detected:    bool
    concept_drift_detected: bool
    reference_faithfulness: float
    current_faithfulness:   float
    severity: str
    details:  dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def drift_detected(self) -> bool:
        return self.data_drift_detected or self.concept_drift_detected

    def to_dict(self) -> dict:
        return {
            "data_drift_score":       round(self.data_drift_score, 4),
            "concept_drift_score":    round(self.concept_drift_score, 4),
            "data_drift_detected":    self.data_drift_detected,
            "concept_drift_detected": self.concept_drift_detected,
            "severity":               self.severity,
            "timestamp":              self.timestamp,
        }


def _compute_js_divergence(ref: list[float], cur: list[float]) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    try:
        import numpy as np
        from scipy.spatial.distance import jensenshannon
        ref_arr = np.array(ref)
        cur_arr = np.array(cur)
        ref_h, bins = np.histogram(ref_arr, bins=50, density=True)
        cur_h, _    = np.histogram(cur_arr, bins=bins, density=True)
        ref_h = (ref_h + 1e-10) / (ref_h + 1e-10).sum()
        cur_h = (cur_h + 1e-10) / (cur_h + 1e-10).sum()
        return float(jensenshannon(ref_h, cur_h))
    except Exception as exc:
        logger.error("JS divergence computation failed: %s", exc)
        return 0.0


def _get_rolling_faithfulness(
    cfg: EnvConfig,
    days: int = 7,
) -> tuple[float, float]:
    """
    Fetch faithfulness scores from MLflow.
    Returns (reference_mean, current_mean) over rolling window.
    """
    try:
        client = get_mlflow_client(cfg)
        now = datetime.utcnow()
        week_ago = now - timedelta(days=days)
        two_weeks_ago = now - timedelta(days=days * 2)
        runs = client.search_runs(
            experiment_names=["ragas-evaluation"],
            filter_string="tags.pipeline = 'P09-RAGAS'",
            order_by=["start_time DESC"],
            max_results=50,
        )
        reference_scores = []
        current_scores   = []
        for run in runs:
            ts = datetime.utcfromtimestamp(run.info.start_time / 1000)
            faith = run.data.metrics.get("faithfulness", None)
            if faith is None:
                continue
            if week_ago <= ts <= now:
                current_scores.append(faith)
            elif two_weeks_ago <= ts < week_ago:
                reference_scores.append(faith)
        ref_mean = sum(reference_scores) / len(reference_scores) if reference_scores else 0.0
        cur_mean = sum(current_scores) / len(current_scores) if current_scores else ref_mean
        return ref_mean, cur_mean
    except Exception as exc:
        logger.error("Rolling faithfulness fetch failed: %s", exc)
        return 0.0, 0.0


def run_drift_check(
    reference_embeddings: list[float] | None = None,
    current_embeddings:   list[float] | None = None,
    cfg: EnvConfig | None = None,
) -> DriftReport:
    """
    Run complete drift check: data drift + concept drift.
    If embeddings not provided, skips data drift check.
    """
    cfg = cfg or EnvConfig()

    # Data drift
    js_div = 0.0
    if reference_embeddings and current_embeddings:
        js_div = _compute_js_divergence(reference_embeddings, current_embeddings)
    data_drift = js_div > JS_DIV_THRESHOLD

    # Concept drift
    ref_faith, cur_faith = _get_rolling_faithfulness(cfg)
    concept_drop = 0.0
    if ref_faith > 0:
        concept_drop = (ref_faith - cur_faith) / ref_faith
    concept_drift = concept_drop >= CONCEPT_DRIFT_DROP_PCT

    # Severity
    if data_drift and concept_drift:
        severity = "HIGH"
    elif data_drift or concept_drift:
        severity = "MEDIUM"
    else:
        severity = "NONE"

    report = DriftReport(
        data_drift_score=js_div,
        concept_drift_score=concept_drop,
        data_drift_detected=data_drift,
        concept_drift_detected=concept_drift,
        reference_faithfulness=ref_faith,
        current_faithfulness=cur_faith,
        severity=severity,
        details={
            "js_div_threshold":        JS_DIV_THRESHOLD,
            "concept_drop_threshold":  CONCEPT_DRIFT_DROP_PCT,
        },
    )
    if report.drift_detected:
        logger.warning("DRIFT DETECTED: severity=%s js_div=%.3f concept_drop=%.1f%%",
                       severity, js_div, concept_drop * 100)
    else:
        logger.info("No drift: js_div=%.3f concept_drop=%.1f%%",
                    js_div, concept_drop * 100)
    return report


def alert_if_needed(
    report: DriftReport,
    slack_webhook_url: str = "",
    p11_webhook_url: str = "",
) -> None:
    """Send Slack alert and optionally trigger P11 retraining."""
    if not report.drift_detected:
        return
    try:
        import requests
        msg = (
            f":warning: *Drift Detected* — severity={report.severity}\n"
            f"Data drift (JS-div): {report.data_drift_score:.3f} "
            f"(threshold={JS_DIV_THRESHOLD})\n"
            f"Concept drift (faithfulness drop): "
            f"{report.concept_drift_score:.1%}\n"
            f"Current faithfulness: {report.current_faithfulness:.3f} "
            f"(was {report.reference_faithfulness:.3f})\n"
            f"Action: P11 Automated Retraining triggered"
        )
        if slack_webhook_url and slack_webhook_url.startswith("https://"):
            requests.post(slack_webhook_url, json={"text": msg}, timeout=10)
        if p11_webhook_url and p11_webhook_url.startswith("http"):
            requests.post(p11_webhook_url,
                          json={"trigger": "drift_alert",
                                "severity": report.severity,
                                "report": report.to_dict()},
                          timeout=15)
            logger.info("P11 retraining triggered")
    except Exception as exc:
        logger.error("Drift alert failed: %s", exc)