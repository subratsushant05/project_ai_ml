FROM python:3.11-slim

# Run as a non-root user.
RUN useradd --create-home --shell /usr/sbin/nologin appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent_evals/ agent_evals/
COPY datasets/ datasets/

RUN mkdir -p /app/reports && chown -R appuser:appuser /app
USER appuser

# Evaluate both bundled agents and write reports to /app/reports.
CMD ["python", "-m", "agent_evals.demo", "--output", "/app/reports"]
