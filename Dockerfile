# Image-generation server (diffusers + FastAPI), used by docker-compose.yml.
# install.sh embeds an equivalent Dockerfile/server.py inline for the curl|sh path —
# keep server.py here in sync with the heredoc in install.sh.
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir -U \
        diffusers transformers accelerate safetensors sentencepiece \
        protobuf fastapi "uvicorn[standard]" pillow hf_transfer

COPY server.py /app/server.py
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
