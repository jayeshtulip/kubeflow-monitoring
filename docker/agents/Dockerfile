FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl kubectl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    httpx==0.27.0 qdrant-client==1.8.0 sentence-transformers==2.7.0 \
    psycopg2-binary==2.9.9 redis==5.0.4 boto3==1.34.69 \
    mlflow==2.12.2 langchain==0.2.0 langgraph==0.1.1 langsmith==0.1.77 \
    prometheus-client==0.20.0 pydantic==2.7.1 kubernetes==26.1.0

COPY src/ src/
COPY pipelines/components/shared/ pipelines/components/shared/
COPY pipelines/__init__.py pipelines/__init__.py

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser
