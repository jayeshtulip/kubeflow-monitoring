import requests
base = 'http://localhost:8080'

# Get P09 v3 pipeline ID
r = requests.get(f'{base}/apis/v1beta1/pipelines')
pipeline_id = ''
for p in r.json().get('pipelines', []):
    if 'P09-RAGAS-Evaluation-v3' in p['name']:
        pipeline_id = p['id']
        print('Found:', p['name'], pipeline_id)

payload = {
    'name': 'p09-ragas-run-5',
    'pipeline_spec': {
        'pipeline_id': pipeline_id,
        'parameters': [
            {'name': 'qa_limit', 'value': '10'},
            {'name': 'postgres_host', 'value': 'llm-platform-prod-postgres.c2xig0uywkrb.us-east-1.rds.amazonaws.com'},
            {'name': 'postgres_password', 'value': 'Llmplatform2026'},
            {'name': 'qdrant_host', 'value': 'qdrant.llm-platform-prod.svc.cluster.local'},
            {'name': 'ollama_base_url', 'value': 'http://ollama-service.llm-platform-prod.svc.cluster.local:11434'},
            {'name': 'mlflow_tracking_uri', 'value': 'http://172.20.232.117:80'}
        ]
    },
    'resource_references': [{'key': {'type': 'EXPERIMENT',
        'id': '21487485-55f3-4529-8c66-90f5710c8e4e'}, 'relationship': 'OWNER'}]
}
r2 = requests.post(f'{base}/apis/v1beta1/runs', json=payload, timeout=10)
print('Run status:', r2.status_code)
print('Run ID:', r2.json().get('run', {}).get('id', ''))
