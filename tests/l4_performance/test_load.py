"""
L4 Load test — Locust user simulation.
Run with:
  locust -f tests/l4_performance/test_load.py --host=http://localhost:8000
  locust -f tests/l4_performance/test_load.py --host=$API_URL -u 10 -r 2 --run-time 60s
"""
from locust import HttpUser, task, between, events
import json, random

SIMPLE_QUERIES = [
    "What is the leave policy for sick days?",
    "Who is the oncall engineer this week?",
    "What are the RDS connection pool limits?",
    "How do I check the status of a Kubernetes deployment?",
    "What is the process for requesting infrastructure access?",
]

COMPLEX_QUERIES = [
    "Why is my EKS payment service timing out intermittently?",
    "Root cause analysis: pods evicted in payment namespace at 14:20",
    "Database connection pool exhausted after CronJob deployment",
]


class LLMPlatformUser(HttpUser):
    """
    Simulates real user traffic: 70% simple, 25% react, 5% PEC.
    Think time: 1-3 seconds between requests.
    """
    wait_time = between(1, 3)
    host = "http://localhost:8000"

    @task(70)
    def simple_query(self):
        query = random.choice(SIMPLE_QUERIES)
        with self.client.post(
            "/api/query",
            json={"query": query, "workflow": "simple"},
            catch_response=True,
            name="/api/query [simple]",
        ) as r:
            if r.status_code == 200:
                data = r.json()
                if not data.get("response"):
                    r.failure("Empty response")
            else:
                r.failure(f"Status {r.status_code}")

    @task(25)
    def react_query(self):
        query = random.choice(SIMPLE_QUERIES)
        with self.client.post(
            "/api/query",
            json={"query": query, "workflow": "react"},
            catch_response=True,
            name="/api/query [react]",
        ) as r:
            if r.status_code != 200:
                r.failure(f"Status {r.status_code}")

    @task(5)
    def pec_query(self):
        query = random.choice(COMPLEX_QUERIES)
        with self.client.post(
            "/api/query",
            json={"query": query},
            catch_response=True,
            name="/api/query [pec]",
            timeout=90,
        ) as r:
            if r.status_code != 200:
                r.failure(f"Status {r.status_code}")

    @task(10)
    def health_check(self):
        self.client.get("/health", name="/health")