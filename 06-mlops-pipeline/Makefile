.PHONY: install pipeline drift-sim serve test lint mlflow-ui docker-build docker-up clean

install:
	pip install -r requirements.txt
	pip install pytest ruff

pipeline:
	python -m mlops_pipeline.run_pipeline

drift-sim:
	python -m mlops_pipeline.simulate_drift

serve:
	uvicorn mlops_pipeline.serving.app:app --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	ruff check .

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

docker-build:
	docker build -t mlops-pipeline .

docker-up:
	docker compose up --build

clean:
	rm -rf mlruns mlruns.db artifacts .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
