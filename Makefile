.PHONY: install format format-check lint type-check test security validate-config validate-docs generate-synthetic-data validate-dataset verify-synthetic-data generate-dicom-fixtures discover-dicom validate-dicom-fixtures verify-dicom-ingestion quality-check-dicom verify-dicom-quality quality clean

PYTHON ?= python3
SYNTHETIC_DATA_DIR ?= data/synthetic/generated
DICOM_FIXTURE_DIR ?= data/dicom/fixtures
DICOM_DEID_DIR ?= data/dicom/deidentified
DICOM_AUDIT_PATH ?= data/dicom/audit/audit.json
DICOM_QUALITY_DIR ?= data/dicom/quality

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

generate-dicom-fixtures:
	$(PYTHON) -m medical_imaging_platform generate-dicom-fixtures --output-dir $(DICOM_FIXTURE_DIR) --overwrite

discover-dicom:
	$(PYTHON) -m medical_imaging_platform discover-dicom $(DICOM_FIXTURE_DIR)

validate-dicom-fixtures:
	$(PYTHON) -m medical_imaging_platform validate-dicom $(DICOM_FIXTURE_DIR) --require-pixel-data

verify-dicom-ingestion: generate-dicom-fixtures discover-dicom validate-dicom-fixtures
	$(PYTHON) -m medical_imaging_platform deidentify-dicom $(DICOM_FIXTURE_DIR) --output-dir $(DICOM_DEID_DIR) --audit-path $(DICOM_AUDIT_PATH) --overwrite
	$(PYTHON) -m medical_imaging_platform validate-dicom $(DICOM_DEID_DIR) --require-pixel-data

quality-check-dicom:
	$(PYTHON) -m medical_imaging_platform quality-report-dicom $(DICOM_DEID_DIR) --output-dir $(DICOM_QUALITY_DIR) --full-pixel-validation --overwrite

verify-dicom-quality: verify-dicom-ingestion quality-check-dicom
	$(PYTHON) -m medical_imaging_platform quality-check-dicom $(DICOM_DEID_DIR) --full-pixel-validation

quality: format-check lint type-check validate-config validate-docs test security

clean:
	rm -rf .coverage coverage.xml .mypy_cache .pytest_cache .ruff_cache src/medical_imaging_platform/__pycache__ src/medical_imaging_platform/deidentification/__pycache__ src/medical_imaging_platform/ingestion/__pycache__ src/medical_imaging_platform/synthetic/__pycache__ src/medical_imaging_platform/utils/__pycache__ tests/__pycache__
