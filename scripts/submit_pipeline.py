from kfp import client

c = client.Client(host='http://localhost:8080')

# Use existing pipeline ID
pipeline_id = '5c4899d6-ba62-4724-a8be-b9929d44d179'
experiment_id = '21487485-55f3-4529-8c66-90f5710c8e4e'

run = c.create_run_from_pipeline_package(
    pipeline_file='test_pipeline.yaml',
    arguments={'name': 'LLM-Platform-Prod'},
    run_name='test-run-2',
    experiment_name='test',
)
print('Run submitted:', run.run_id)
