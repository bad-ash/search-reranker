FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.hf
ENV HF_HUB_CACHE=/app/.hf/hub
ENV SENTENCE_TRANSFORMERS_HOME=/app/.hf/sentence_transformers

COPY pyproject.toml README.md /app/
COPY service /app/service
COPY training /app/training
COPY scripts /app/scripts
COPY artifacts /app/artifacts

RUN pip install --no-cache-dir .
RUN python /app/scripts/preload_models.py

EXPOSE 8000

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
