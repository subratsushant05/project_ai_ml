FROM python:3.11-slim

# Install dependencies first for better layer caching.
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mlops_pipeline/ mlops_pipeline/

# Run as a non-root user; /app must stay writable for mlruns.db + artifacts.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Train + promote a first model at container start if none exists, then serve.
CMD ["sh", "-c", "python -m mlops_pipeline.run_pipeline && uvicorn mlops_pipeline.serving.app:app --host 0.0.0.0 --port 8000"]
