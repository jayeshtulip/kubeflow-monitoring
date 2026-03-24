"""kubectl / EKS tool for the Executor agent."""
from __future__ import annotations
import subprocess
import json
import shutil
from pipelines.components.shared.base import get_logger

logger = get_logger(__name__)

_ALLOWED_VERBS = {"get", "describe", "logs", "top"}
_ALLOWED_RESOURCES = {
    "pods", "pod", "po",
    "nodes", "node", "no",
    "events", "event",
    "deployments", "deployment", "deploy",
    "services", "service", "svc",
    "configmaps", "configmap", "cm",
    "cronjobs", "cronjob",
    "hpa",
}


def run_kubectl(
    verb: str,
    resource: str,
    name: str = "",
    namespace: str = "",
    extra_args: list[str] | None = None,
) -> dict:
    """
    Run a safe kubectl command (read-only verbs only).
    Returns dict with stdout, stderr, returncode.
    """
    if verb not in _ALLOWED_VERBS:
        return {"error": f"Verb '{verb}' not allowed. Allowed: {_ALLOWED_VERBS}",
                "tool": "kubectl"}
    if resource.lower() not in _ALLOWED_RESOURCES:
        return {"error": f"Resource '{resource}' not allowed. Allowed: {_ALLOWED_RESOURCES}",
                "tool": "kubectl"}
    if not shutil.which("kubectl"):
        return {"error": "kubectl not found in PATH", "tool": "kubectl"}
    cmd = ["kubectl", verb, resource]
    if name:
        cmd.append(name)
    if namespace:
        cmd.extend(["-n", namespace])
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(["-o", "json"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                data = {"raw": result.stdout[:2000]}
        else:
            data = {"error": result.stderr[:500]}
        data["tool"] = "kubectl"
        data["command"] = " ".join(cmd)
        logger.info("kubectl %s", " ".join(cmd[1:]))
        return data
    except subprocess.TimeoutExpired:
        return {"error": "kubectl timeout after 30s", "tool": "kubectl"}
    except Exception as exc:
        logger.error("kubectl error: %s", exc)
        return {"error": str(exc), "tool": "kubectl"}


def get_pod_events(namespace: str = "default") -> dict:
    return run_kubectl("get", "events", namespace=namespace,
                      extra_args=["--sort-by=.lastTimestamp"])


def get_pods(namespace: str = "default") -> dict:
    return run_kubectl("get", "pods", namespace=namespace)


def describe_pod(pod_name: str, namespace: str = "default") -> dict:
    return run_kubectl("describe", "pod", name=pod_name, namespace=namespace)