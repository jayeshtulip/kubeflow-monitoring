import requests

base = 'http://localhost:8080'

payload = {
    'name': 'test-run-v1',
    'pipeline_spec': {
        'pipeline_id': '5c4899d6-ba62-4724-a8be-b9929d44d179'
    },
    'resource_references': [{
        'key': {
            'type': 'EXPERIMENT',
            'id': '21487485-55f3-4529-8c66-90f5710c8e4e'
        },
        'relationship': 'OWNER'
    }]
}

r = requests.post(f'{base}/apis/v1beta1/runs', json=payload, timeout=10)
print('Status:', r.status_code)
print('Response:', r.text[:500])
