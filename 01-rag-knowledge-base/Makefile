.PHONY: install dev test lint demo run docker-build docker-run

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

test:
	pytest -q

lint:
	ruff check .

demo:
	python -m rag_kb.demo

run:
	uvicorn rag_kb.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t rag-kb .

docker-run:
	docker run --rm -p 8000:8000 rag-kb
