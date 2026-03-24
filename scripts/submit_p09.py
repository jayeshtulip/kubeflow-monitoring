import requests
base = 'http://localhost:8080'
payload = {
    'name': 'p09-ragas-run-2',
    'pipeline_spec': {'pipeline_id': '3697d2c6-9f11-47a1-8800-9fafd9d96abb'},
    'resource_references': [{'key': {'type': 'EXPERIMENT',
        'id': '21487485-55f3-4529-8c66-90f5710c8e4e'}, 'relationship': 'OWNER'}]
}
r = requests.post(f'{base}/apis/v1beta1/runs', json=payload, timeout=10)
print('Run status:', r.status_code)
print('Run ID:', r.json().get('run', {}).get('id', ''))
