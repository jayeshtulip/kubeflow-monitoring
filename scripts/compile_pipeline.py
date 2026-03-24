from kfp import dsl, compiler
from kfp.dsl import component

@component(base_image='python:3.11-slim')
def say_hello(name: str) -> str:
    message = 'Hello ' + name
    print(message)
    return message

@dsl.pipeline(name='test-hello-pipeline')
def hello_pipeline(name: str = 'LLM-Platform'):
    say_hello(name=name)

compiler.Compiler().compile(hello_pipeline, 'test_pipeline.yaml')
print('Compiled successfully')
