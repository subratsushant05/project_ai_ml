#!/bin/sh
# Train a model on first start if no artifacts were baked into the image,
# then serve the API.
set -e

if [ ! -f "${CHURN_ARTIFACTS_DIR:-artifacts}/model.joblib" ]; then
    echo "No model artifacts found - running training once..."
    python -m churn_pipeline.train
fi

exec uvicorn churn_pipeline.api:app --host 0.0.0.0 --port 8000
