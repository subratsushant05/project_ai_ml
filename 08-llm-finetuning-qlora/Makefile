# qlora_tune developer tasks. Light targets run anywhere; train/merge need a GPU.

PYTHON ?= python3

.PHONY: install install-train demo dry-run test lint train merge clean

install:            ## Install light (CPU-only) dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-train:      ## Install heavy GPU training stack (CUDA required)
	$(PYTHON) -m pip install -r requirements.txt -r requirements-train.txt

demo:               ## Run the end-to-end light-path demo
	$(PYTHON) -m qlora_tune.demo

dry-run:            ## Print the training plan for the 8B config (no heavy deps)
	$(PYTHON) -m qlora_tune.train --config configs/llama-3.1-8b.yaml --dry-run

test:               ## Run the test suite (light deps + pytest)
	$(PYTHON) -m pytest

lint:               ## Ruff lint
	$(PYTHON) -m ruff check .

train:              ## Real QLoRA training run (GPU + requirements-train.txt)
	$(PYTHON) -m qlora_tune.train --config configs/llama-3.1-8b.yaml \
		--train-file demo_output/train.jsonl --val-file demo_output/val.jsonl

merge:              ## Merge the trained adapter into the base model (GPU box)
	$(PYTHON) -m qlora_tune.merge --base meta-llama/Llama-3.1-8B-Instruct \
		--adapter outputs/llama31-8b-helpdesk --out outputs/merged

clean:
	rm -rf demo_output outputs .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
