# Serve the churn prediction API.
#
# The image trains a model at container start if no artifacts are baked in.
# For reproducible deploys, run `python -m churn_pipeline.train` before
# building so artifacts/ is copied into the image.

FROM python:3.11-slim

# libgomp1 is LightGBM's OpenMP runtime dependency.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY churn_pipeline ./churn_pipeline
COPY docker/entrypoint.sh ./entrypoint.sh
# Baked-in artifacts are optional; the entrypoint trains if they are absent.
COPY artifact[s] ./artifacts

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app \
    && chmod +x entrypoint.sh
USER appuser

ENV CHURN_ARTIFACTS_DIR=/app/artifacts
EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
