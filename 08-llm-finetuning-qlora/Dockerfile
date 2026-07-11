# Light image: runs the full CPU demo (data pipeline, dry-run plan, evaluation).
# No GPU, no torch — ~200 MB instead of ~10 GB.
#
#   docker build -t qlora-tune-demo .
#   docker run --rm qlora-tune-demo

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY qlora_tune/ qlora_tune/
COPY configs/ configs/
COPY sample_data/ sample_data/

CMD ["python", "-m", "qlora_tune.demo", "--out", "/tmp/demo_output"]
