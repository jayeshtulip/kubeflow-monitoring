from kfp import dsl, client
from kfp.dsl import component

@component(base_image='python:3.11-slim')
def say_hello(name: str) -> str:
    message = f'Hello {name} from KFP on EKS!'
    print(message)
    return message

@dsl.pipeline(name='test-hello-pipeline')
def hello_pipeline(name: str = 'LLM-Platform'):
    say_hello(name=name)

if __name__ == '__main__':
    c = client.Client(host='http://localhost:8080')
    run = c.create_run_from_pipeline_func(
        hello_pipeline,
        arguments={'name': 'LLM-Platform-Prod'},
        run_name='test-run-1',
        experiment_name='test',
    )
    print(f'Run submitted: {run.run_id}')
    print(f'Monitor: http://localhost:8080/apis/v2beta1/runs/{run.run_id}')
