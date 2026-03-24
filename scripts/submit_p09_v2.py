import requests
base = 'http://localhost:8080'

# Get P09 pipeline ID
r = requests.get(f'{base}/apis/v1beta1/pipelines')
pipeline_id = ''
for p in r.json().get('pipelines', []):
    if 'P09' in p['name'] or 'ragas' in p['name'].lower():
        pipeline_id = p['id']
        print('Found:', p['name'], pipeline_id)

if not pipeline_id:
    # Upload fresh
    with open('p09_pipeline.yaml', 'rb') as f:
        r2 = requests.post(f'{base}/apis/v1beta1/pipelines/upload',
            files={'uploadfile': ('p09_pipeline.yaml', f, 'application/yaml')},
            params={'name': 'P09-RAGAS-Evaluation-v2'})
    pipeline_id = r2.json().get('id', '')
    print('Uploaded:', pipeline_id)

payload = {'name': 'p09-ragas-run-3',
    'pipeline_spec': {'pipeline_id': pipeline_id},
    'resource_references': [{'key': {'type': 'EXPERIMENT',
        'id': '21487485-55f3-4529-8c66-90f5710c8e4e'}, 'relationship': 'OWNER'}]}
r3 = requests.post(f'{base}/apis/v1beta1/runs', json=payload, timeout=10)
print('Run status:', r3.status_code)
print('Run ID:', r3.json().get('run', {}).get('id', ''))
