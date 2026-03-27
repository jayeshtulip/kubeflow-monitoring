from locust import HttpUser, task, between
import json, random

PROMPTS = [
    "What is Kubernetes in one sentence?",
    "Explain pod evictions in EKS briefly.",
    "What is XGBoost scale_pos_weight?",
    "How does PagedAttention work in vLLM?",
    "What is RAGAS faithfulness metric?",
    "How does ArgoCD GitOps work?",
    "What is Jensen-Shannon divergence?",
    "Explain LangGraph PEC workflow briefly.",
    "What is RDS connection pool exhaustion?",
    "How does NVIDIA Triton FIL backend work?",
]

MODEL = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

class VLLMUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://localhost:8000"

    @task(4)
    def inference(self):
        prompt = random.choice(PROMPTS)
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0,
        }
        with self.client.post(
            "/v1/completions",
            json=payload,
            timeout=60,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                tokens = data.get("usage", {}).get("completion_tokens", 0)
                response.success()
                response.meta = {"tokens": tokens}
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def health_check(self):
        with self.client.get(
            "/health",
            timeout=10,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
