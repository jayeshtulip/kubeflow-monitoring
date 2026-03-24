import requests
base = 'http://localhost:8080'
with open('p01_pipeline.yaml', 'rb') as f:
    r = requests.post(f'{base}/apis/v1beta1/pipelines/upload',
        files={'uploadfile': ('p01_pipeline.yaml', f, 'application/yaml')},
        params={'name': 'P01-Model-Evaluation-v6'})
pipeline_id = r.json().get('id', '')
print('Pipeline ID:', pipeline_id)
payload = {'name': 'p01-run-v6',
    'pipeline_spec': {'pipeline_id': pipeline_id},
    'resource_references': [{'key': {'type': 'EXPERIMENT',
    'id': '21487485-55f3-4529-8c66-90f5710c8e4e'}, 'relationship': 'OWNER'}]}
r2 = requests.post(f'{base}/apis/v1beta1/runs', json=payload, timeout=10)
print('Run ID:', r2.json().get('run', {}).get('id', ''))
