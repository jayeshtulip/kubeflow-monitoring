import requests
base = 'http://localhost:8080'

r = requests.get(f'{base}/apis/v1beta1/pipelines')
pipeline_id = ''
for p in r.json().get('pipelines', []):
    if 'P09-RAGAS-Evaluation-v9' in p['name']:
        pipeline_id = p['id']
        print('Found:', p['name'], pipeline_id)

payload = {
    'name': 'p09-ragas-run-13-nocache',
    'pipeline_spec': {'pipeline_id': pipeline_id},
    'resource_references': [{'key': {'type': 'EXPERIMENT',
        'id': '21487485-55f3-4529-8c66-90f5710c8e4e'}, 'relationship': 'OWNER'}],
    'service_account': 'pipeline-runner',
    'max_concurrent_node_count': '1'
}
r2 = requests.post(f'{base}/apis/v1beta1/runs', json=payload, timeout=10)
print('Run status:', r2.status_code)
run_id = r2.json().get('run', {}).get('id', '')
print('Run ID:', run_id)

# Disable cache for this run
requests.patch(f'{base}/apis/v1beta1/runs/{run_id}',
    json={'resource_references': [], 'disable_cache': True}, timeout=5)
