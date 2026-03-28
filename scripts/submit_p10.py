import requests, time

base = 'http://localhost:8080'
ts = int(time.time())

with open('p10_pipeline.yaml', 'rb') as f:
    r = requests.post(f'{base}/apis/v1beta1/pipelines/upload',
        files={'uploadfile': ('p10_pipeline.yaml', f, 'application/yaml')},
        params={'name': f'P10-DVC-Reproducibility-{ts}'})
print('Upload:', r.status_code)
pipeline_id = r.json().get('id', '')
print('Pipeline ID:', pipeline_id)

if pipeline_id:
    payload = {
        'name': f'p10-dvc-run-{ts}',
        'pipeline_spec': {
            'pipeline_id': pipeline_id,
            'parameters': [
                {'name': 'stage', 'value': 'validate_golden_qa'}
            ]
        },
        'resource_references': [{'key': {'type': 'EXPERIMENT',
            'id': '21487485-55f3-4529-8c66-90f5710c8e4e'}, 'relationship': 'OWNER'}]
    }
    r2 = requests.post(f'{base}/apis/v1beta1/runs', json=payload, timeout=10)
    print('Run status:', r2.status_code)
    print('Run ID:', r2.json().get('run', {}).get('id', ''))
