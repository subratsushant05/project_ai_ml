FROM python:3.11-slim

# libgomp1 is required by LightGBM at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /usr/sbin/nologin appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ts_forecast/ ts_forecast/

USER appuser
ENV TSF_OUTPUT_DIR=/tmp/output

CMD ["python", "-m", "ts_forecast.demo"]
