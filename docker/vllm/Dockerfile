FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HUGGINGFACE_HUB_CACHE=/models/hub \
    TRANSFORMERS_CACHE=/models/hub

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    vllm==0.4.2 \
    transformers==4.40.1 \
    accelerate==0.29.3 \
    huggingface_hub==0.22.2

VOLUME ["/models"]

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/models/Llama-2-7b-chat-hf-GPTQ", \
     "--quantization", "gptq", \
     "--dtype", "float16", \
     "--max-model-len", "4096", \
     "--port", "8080", \
     "--host", "0.0.0.0", \
     "--served-model-name", "llama2-7b-gptq", \
     "--tensor-parallel-size", "1"]
