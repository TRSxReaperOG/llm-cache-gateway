FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "llm_cache_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
