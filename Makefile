# Single entry point for the common workflows. Run `make help` for the list.

PY ?= uv run

.DEFAULT_GOAL := help
.PHONY: help install train cv test lint format notebook clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Create the environment from uv.lock (pipeline + notebook + dev tools)
	uv sync --frozen --all-extras
	uv run pre-commit install

train:  ## Train the final model and write submissions/final_ensemble.csv (~30 s)
	$(PY) classical-music-ml

cv:  ## Same, and also print the 15-fold cross-validated AUROC
	$(PY) classical-music-ml --cv

test:  ## Run the tests (they use a synthetic copy of the data, no download needed)
	$(PY) pytest

lint:  ## Lint and check formatting
	$(PY) ruff check .
	$(PY) ruff format --check .

format:  ## Auto-fix lint issues and format the code
	$(PY) ruff check --fix .
	$(PY) ruff format .

notebook:  ## Open the full analysis notebook
	$(PY) jupyter notebook notebooks/classical_music_ml.ipynb

clean:  ## Remove caches and run folders
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov runs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
