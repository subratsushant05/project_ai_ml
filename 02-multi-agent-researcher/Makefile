.PHONY: install install-dev test lint demo docker-build docker-run clean

PYTHON ?= python3

install:  ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install  ## Install runtime + development tooling
	$(PYTHON) -m pip install pytest ruff

test:  ## Run the offline test suite
	$(PYTHON) -m pytest

lint:  ## Static checks
	ruff check .

demo:  ## Run the offline end-to-end demo
	$(PYTHON) -m agent_researcher.demo "How do transformer models work?"

docker-build:  ## Build the container image
	docker build -t agent-researcher .

docker-run:  ## Run the demo inside the container
	docker run --rm agent-researcher

clean:  ## Remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache report.md
	find . -type d -name __pycache__ -exec rm -rf {} +
