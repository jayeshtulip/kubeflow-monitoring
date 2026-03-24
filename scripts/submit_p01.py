import requests

base = 'http://localhost:8080'

payload = {
    'name': 'p01-model-evaluation-run-3',
    'pipeline_spec': {
        'pipeline_id': 'a5b6bcd4-3b08-4058-a623-36be8bd95495',
        'parameters': [
            {'name': 'mlflow_tracking_uri', 'value': 'http://mlflow.mlflow.svc.cluster.local:80'}
        ]
    },
    'resource_references': [{
        'key': {'type': 'EXPERIMENT', 'id': '21487485-55f3-4529-8c66-90f5710c8e4e'},
        'relationship': 'OWNER'
    }]
}
r = requests.post(f'{base}/apis/v1beta1/runs', json=payload, timeout=10)
print('Run status:', r.status_code)
print('Run ID:', r.json().get('run', {}).get('id', ''))
