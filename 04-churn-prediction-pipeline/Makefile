.PHONY: install train demo test lint serve docker-build docker-run clean

install:
	pip install -r requirements.txt
	pip install pytest ruff httpx

train:
	python -m churn_pipeline.train

demo:
	python -m churn_pipeline.demo

test:
	pytest

lint:
	ruff check .

serve:
	uvicorn churn_pipeline.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t churn-pipeline .

docker-run:
	docker run --rm -p 8000:8000 churn-pipeline

clean:
	rm -rf artifacts .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
