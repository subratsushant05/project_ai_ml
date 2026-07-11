.PHONY: install dev test lint demo docker clean

install:
	pip install -r requirements.txt

dev: install
	pip install pytest ruff

test:
	pytest -q

lint:
	ruff check .

demo:
	python -m agent_evals.demo --output reports

docker:
	docker build -t agent-evals .
	docker run --rm agent-evals

clean:
	rm -rf reports .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
