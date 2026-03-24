import requests

base = 'http://localhost:8080'

# Upload P04 pipeline
import os
with open('p04_pipeline.yaml', 'rb') as f:
    r = requests.post(
        f'{base}/apis/v1beta1/pipelines/upload',
        files={'uploadfile': ('p04_pipeline.yaml', f, 'application/yaml')},
        params={'name': 'P04-Quality-Monitoring'}
    )
print('Upload status:', r.status_code)
pipeline_id = r.json().get('id', '')
print('Pipeline ID:', pipeline_id)

# Submit run
payload = {
    'name': 'p04-quality-monitoring-run-1',
    'pipeline_spec': {'pipeline_id': pipeline_id},
    'resource_references': [{
        'key': {'type': 'EXPERIMENT', 'id': '21487485-55f3-4529-8c66-90f5710c8e4e'},
        'relationship': 'OWNER'
    }]
}
r2 = requests.post(f'{base}/apis/v1beta1/runs', json=payload, timeout=10)
print('Run status:', r2.status_code)
print('Run ID:', r2.json().get('run', {}).get('id', ''))
