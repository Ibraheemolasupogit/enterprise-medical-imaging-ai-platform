.PHONY: install format format-check lint type-check test security validate-config validate-docs generate-synthetic-data validate-dataset verify-synthetic-data quality clean

PYTHON ?= python3
SYNTHETIC_DATA_DIR ?= data/synthetic/generated

install:
	$(PYTHON) -m pip install -e ".[dev]"

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

lint:
	$(PYTHON) -m ruff check .

type-check:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest

security:
	$(PYTHON) -m bandit -c pyproject.toml -r src

validate-config:
	$(PYTHON) -m medical_imaging_platform validate-config

validate-docs:
	$(PYTHON) scripts/validate_docs.py

generate-synthetic-data:
	$(PYTHON) -m medical_imaging_platform generate-synthetic-data --output-dir $(SYNTHETIC_DATA_DIR) --overwrite

validate-dataset:
	$(PYTHON) -m medical_imaging_platform validate-dataset $(SYNTHETIC_DATA_DIR)

verify-synthetic-data: generate-synthetic-data validate-dataset

quality: format-check lint type-check validate-config validate-docs test security

clean:
	rm -rf .coverage coverage.xml .mypy_cache .pytest_cache .ruff_cache src/medical_imaging_platform/__pycache__ src/medical_imaging_platform/synthetic/__pycache__ src/medical_imaging_platform/utils/__pycache__ tests/__pycache__
