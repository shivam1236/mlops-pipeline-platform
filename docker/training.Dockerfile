FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[training]"

COPY src/ ./src/
COPY pipelines/ ./pipelines/
COPY configs/ ./configs/

CMD ["python", "-m", "pipelines.training_pipeline"]
