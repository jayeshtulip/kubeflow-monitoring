FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc libpq-dev git && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.111.0 uvicorn==0.29.0 httpx==0.27.0 \
    qdrant-client==1.8.0 sentence-transformers==2.7.0 \
    psycopg2-binary==2.9.9 redis==5.0.4 boto3==1.34.69 \
    mlflow==2.12.2 prometheus-client==0.20.0 \
    langchain==0.2.0 langgraph==0.1.1 langsmith==0.1.77 \
    pydantic==2.7.1 sqlalchemy==2.0.29

# App source
COPY src/ src/
COPY pipelines/components/shared/ pipelines/components/shared/
COPY pipelines/__init__.py pipelines/__init__.py

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--loop", "uvloop", "--http", "httptools"]
