FROM python:3.11-slim
RUN pip install --no-cache-dir \
    kfp==2.2.0 \
    mlflow==2.12.2 \
    psycopg2-binary==2.9.9 \
    requests==2.32.0 \
    httpx==0.27.0 \
    boto3==1.34.0
RUN pip install --no-cache-dir \
    qdrant-client==1.8.0 \
    sentence-transformers==2.7.0
RUN pip install --no-cache-dir \
    langchain-core==0.1.52 \
    langchain==0.1.20 \
    langchain-community==0.0.38
RUN pip install --no-cache-dir \
    ragas==0.1.0 \
    datasets==2.19.0
