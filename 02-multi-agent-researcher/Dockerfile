# Offline-first demo image: no API keys or network needed at runtime.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    AGENT_CORPUS_PATH=/app/sample_data/corpus.json

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY agent_researcher/ ./agent_researcher/
COPY sample_data/ ./sample_data/

# Run as an unprivileged user; write the report to its home directory.
RUN useradd --create-home --uid 1000 appuser
USER appuser
WORKDIR /home/appuser

ENTRYPOINT ["python", "-m", "agent_researcher.demo"]
CMD ["How do transformer models work?"]
