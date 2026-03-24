import requests
base = 'http://localhost:8080'

pipeline_id = '6ba96734-13f6-4d33-a78c-86cdd9936d21'
print('Using P09-RAGAS-Evaluation-v11:', pipeline_id)

payload = {
    'name': 'p09-ragas-run-21-fresh',
    'pipeline_spec': {
        'pipeline_id': pipeline_id,
        'parameters': [{'name': 'qa_limit', 'value': '2'}]
    },
    'resource_references': [{'key': {'type': 'EXPERIMENT',
        'id': '21487485-55f3-4529-8c66-90f5710c8e4e'}, 'relationship': 'OWNER'}]
}
r2 = requests.post(f'{base}/apis/v1beta1/runs', json=payload, timeout=10)
print('Run status:', r2.status_code)
print('Run ID:', r2.json().get('run', {}).get('id', ''))
