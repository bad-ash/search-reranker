FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY service /app/service
COPY training /app/training
COPY artifacts /app/artifacts
COPY tests /app/tests

RUN pip install --no-cache-dir .

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
