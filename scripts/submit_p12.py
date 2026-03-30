import requests, uuid, os, sys
BASE = os.environ.get("KFP_ENDPOINT", "http://localhost:8080")
RUN_UUID = f"p12-e2e-{uuid.uuid4().hex[:8]}"
print(f"Submitting P12 | uuid={RUN_UUID}")
yaml_path = os.path.join(os.path.dirname(__file__), "..", "p12_pipeline.yaml")
if not os.path.exists(yaml_path):
    yaml_path = os.path.join(os.path.dirname(__file__), "p12_pipeline.yaml")
if not os.path.exists(yaml_path):
    print("ERROR: p12_pipeline.yaml not found. Run compile_p12.py first.")
    sys.exit(1)
with open(yaml_path, "rb") as f:
    up = requests.post(f"{BASE}/apis/v2beta1/pipelines/upload",
        files={"uploadfile": (f"p12-{RUN_UUID}.yaml", f, "application/yaml")},
        data={"name": f"p12-{RUN_UUID}"}, timeout=30)
if not up.ok:
    print(f"Upload failed: {up.status_code} {up.text[:200]}"); sys.exit(1)
pipeline_id = up.json()["pipeline_id"]
print(f"Pipeline uploaded: {pipeline_id}")
run_resp = requests.post(f"{BASE}/apis/v2beta1/runs", json={
    "display_name": f"p12-e2e-{RUN_UUID}",
    "experiment_id": "21487485-55f3-4529-8c66-90f5710c8e4e",
    "pipeline_version_reference": {"pipeline_id": pipeline_id},
    "runtime_config": {"parameters": {"run_uuid": RUN_UUID}, "enable_caching": False},
}, timeout=15)
if not run_resp.ok:
    print(f"Run failed: {run_resp.status_code} {run_resp.text[:200]}"); sys.exit(1)
run_id = run_resp.json()["run_id"]
print(f"Run submitted: {run_id}")
print(f"UI: http://localhost:8080/#/runs/details/{run_id}")
