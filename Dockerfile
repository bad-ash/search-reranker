FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md /app/
COPY service /app/service
COPY training /app/training
COPY artifacts /app/artifacts

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
