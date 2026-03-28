import requests, time, os

base     = os.environ.get("KFP_ENDPOINT", "http://localhost:8080")
ts       = int(time.time())
run_ts   = str(ts)  # passed as param to bust KFP cache

print(f"Submitting P10 to KFP at {base} | cache-bust ts={run_ts}")

with open("p10_pipeline.yaml", "rb") as f:
    r = requests.post(
        f"{base}/apis/v1beta1/pipelines/upload",
        files={"uploadfile": ("p10_pipeline.yaml", f, "application/yaml")},
        params={"name": f"P10-DVC-Reproducibility-{ts}"},
        timeout=30
    )
print(f"Upload: {r.status_code}")
if r.status_code not in (200, 201):
    print(f"Upload failed: {r.text}")
    exit(1)

pipeline_id = r.json().get("id", "")
print(f"Pipeline ID: {pipeline_id}")

# Pass run_ts as a dummy param change to bust KFP fingerprint cache
payload = {
    "name": f"p10-dvc-run-{ts}",
    "pipeline_spec": {
        "pipeline_id": pipeline_id,
        "parameters": [
            {"name": "stage",      "value": "all"},
            {"name": "aws_region", "value": f"us-east-1#{run_ts}"},  # cache buster
        ]
    },
    "resource_references": [{
        "key": {
            "type": "EXPERIMENT",
            "id": "21487485-55f3-4529-8c66-90f5710c8e4e"
        },
        "relationship": "OWNER"
    }]
}

r2 = requests.post(f"{base}/apis/v1beta1/runs", json=payload, timeout=30)
print(f"Run status: {r2.status_code}")
run = r2.json().get("run", {})
run_id   = run.get("id", "")
run_name = run.get("name", "")
print(f"Run ID:   {run_id}")
print(f"Run Name: {run_name}")
print(f"KFP URL:  {base}/#/runs/details/{run_id}")
