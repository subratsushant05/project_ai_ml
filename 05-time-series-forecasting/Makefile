.PHONY: install test lint demo docker-build docker-run clean

install:
	pip install -r requirements.txt
	pip install pytest ruff

test:
	pytest tests/ -q

lint:
	ruff check .

demo:
	python -m ts_forecast.demo

docker-build:
	docker build -t ts-forecast .

docker-run:
	docker run --rm ts-forecast

clean:
	rm -rf output/ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
