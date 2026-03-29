import requests, uuid, os, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--trigger", default="manual",
                    choices=["manual", "p09_gate_fail", "scheduled", "drift_detected"])
args = parser.parse_args()

BASE     = os.environ.get("KFP_ENDPOINT", "http://localhost:8080")
RUN_UUID = f"p11-{args.trigger[:12]}-{uuid.uuid4().hex[:8]}"

print(f"Submitting P11 | trigger={args.trigger} | uuid={RUN_UUID}")

yaml_path = os.path.join(os.path.dirname(__file__), "..", "p11_pipeline.yaml")
if not os.path.exists(yaml_path):
    yaml_path = os.path.join(os.path.dirname(__file__), "p11_pipeline.yaml")
if not os.path.exists(yaml_path):
    print("ERROR: p11_pipeline.yaml not found. Run compile_p11.py first.")
    sys.exit(1)

with open(yaml_path, "rb") as f:
    up_resp = requests.post(
        f"{BASE}/apis/v2beta1/pipelines/upload",
        files={"uploadfile": (f"p11-{RUN_UUID}.yaml", f, "application/yaml")},
        data={"name": f"p11-{RUN_UUID}"},
        timeout=30,
    )

if not up_resp.ok:
    print(f"Upload failed: {up_resp.status_code} {up_resp.text[:300]}")
    sys.exit(1)

pipeline_id = up_resp.json()["pipeline_id"]
print(f"Pipeline uploaded: {pipeline_id}")

EXPERIMENT_ID = "21487485-55f3-4529-8c66-90f5710c8e4e"

run_payload = {
    "display_name": f"p11-auto-retraining-{RUN_UUID}",
    "experiment_id": EXPERIMENT_ID,
    "pipeline_version_reference": {"pipeline_id": pipeline_id},
    "runtime_config": {
        "parameters": {
            "run_uuid": RUN_UUID,
        },
        "enable_caching": False,
    },
}

run_resp = requests.post(f"{BASE}/apis/v2beta1/runs", json=run_payload, timeout=15)
if not run_resp.ok:
    print(f"Run submission failed: {run_resp.status_code} {run_resp.text[:300]}")
    sys.exit(1)

run_id = run_resp.json()["run_id"]
print(f"Run submitted: {run_id}")
print(f"UI: http://localhost:8080/#/runs/details/{run_id}")
print(f"Monitor: kubectl get workflow -n kubeflow --sort-by=.metadata.creationTimestamp | Select-Object -Last 3")

