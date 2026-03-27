import requests
base = 'http://localhost:8080'

with open('p09_pipeline.yaml', 'rb') as f:
    r = requests.post(f'{base}/apis/v1beta1/pipelines/upload',
        files={'uploadfile': ('p09_pipeline.yaml', f, 'application/yaml')},
        params={'name': 'P09-RAGAS-vLLM-v4'})
print('Upload:', r.status_code)
pipeline_id = r.json().get('id', '')
print('Pipeline ID:', pipeline_id)

if not pipeline_id:
    print('Failed:', r.text[:200])
else:
    payload = {
        'name': 'p09-ragas-vllm-run-4',
        'pipeline_spec': {
            'pipeline_id': pipeline_id,
            'parameters': [
                {'name': 'qa_limit', 'value': '2'},
                {'name': 'run_name', 'value': 'ragas-vllm-judge-run-4'}
            ]
        },
        'resource_references': [{'key': {'type': 'EXPERIMENT',
            'id': '21487485-55f3-4529-8c66-90f5710c8e4e'}, 'relationship': 'OWNER'}]
    }
    r2 = requests.post(f'{base}/apis/v1beta1/runs', json=payload, timeout=10)
    print('Run status:', r2.status_code)
    print('Run ID:', r2.json().get('run', {}).get('id', ''))
