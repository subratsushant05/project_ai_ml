.PHONY: install seed serve demo test lint docker-build docker-run clean

install:
	pip install -r requirements.txt
	pip install pytest ruff

seed:
	python -m mcp_analytics.seed --force

serve:
	python -m mcp_analytics.server

demo:
	python -m mcp_analytics.demo

test:
	pytest

lint:
	ruff check .

docker-build:
	docker build -t mcp-analytics .

docker-run:
	docker run -i --rm mcp-analytics

clean:
	rm -rf .pytest_cache .ruff_cache data/*.db
	find . -type d -name __pycache__ -exec rm -rf {} +
