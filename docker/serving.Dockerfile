FROM python:3.10-slim as builder

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/serving/ ./src/serving/

EXPOSE 8080

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
