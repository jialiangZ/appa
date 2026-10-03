PYTHON ?= .venv/bin/python
# generate.py manages its own torch.distributed setup and expects a torchrun
# process group; the Lightning-powered training entry points do not.
TORCHRUN ?= .venv/bin/torchrun
GPUS ?= 1
ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help install sync test lint format bench train-ae train-den generate clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (editable) with all extras into .venv
	uv pip install -e ".[all]" --python $(PYTHON)

sync: ## Recreate the environment exactly from uv.lock
	uv sync --extra all

test: ## Run the test suite
	$(PYTHON) -m pytest tests -q

lint: ## Run all pre-commit hooks (ruff lint + format check)
	$(PYTHON) -m pre_commit run --all-files --config pre-commit.yml

format: ## Auto-format with ruff
	$(PYTHON) -m ruff format src tests tools run_*.py
	$(PYTHON) -m ruff check --fix src tests tools run_*.py

bench: ## Benchmark ERA5Dataset loading modes
	$(PYTHON) src/scripts/bench_dataloader.py

train-ae: ## Train the autoencoder. Usage: make train-ae GPUS=1 ARGS="train.update_steps=2000"
	CUDA_VISIBLE_DEVICES=$${CUDA_VISIBLE_DEVICES:-0} SWANLAB_MODE=$${SWANLAB_MODE:-disabled} \
		APPA_DEVICES=$(GPUS) $(PYTHON) -m appa.launchers ae-train $(ARGS)

train-den: ## Train the denoiser. Usage: make train-den ARGS="ae_run=<id>/0 latent_dump=default"
	CUDA_VISIBLE_DEVICES=$${CUDA_VISIBLE_DEVICES:-0} SWANLAB_MODE=$${SWANLAB_MODE:-disabled} \
		APPA_DEVICES=$(GPUS) $(PYTHON) -m appa.launchers den-train $(ARGS)

generate: ## Generate prior trajectories. Usage: make generate ARGS="<model_path> diffusion.num_steps=64"
	$(TORCHRUN) --standalone --nproc-per-node 1 -m appa.launchers generate $(ARGS)

clean: ## Remove caches and build artifacts
	find . -name __pycache__ -not -path "./.venv/*" -not -path "./artifacts/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .cache build dist *.egg-info
