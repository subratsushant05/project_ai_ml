FROM python:3.11-slim

# Create an unprivileged user; MCP servers do not need root.
RUN useradd --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp_analytics/ mcp_analytics/

# Database and charts live under the app user's home.
ENV MCP_ANALYTICS_DB=/home/appuser/data/ecommerce.db \
    MCP_ANALYTICS_CHART_DIR=/home/appuser/charts \
    PYTHONUNBUFFERED=1

RUN mkdir -p /home/appuser/data /home/appuser/charts \
    && chown -R appuser:appuser /home/appuser /app

USER appuser

# Seed at build time so the container starts instantly.
RUN python -m mcp_analytics.seed

# MCP over stdio: run with `docker run -i` so stdin stays open.
ENTRYPOINT ["python", "-m", "mcp_analytics.server"]
