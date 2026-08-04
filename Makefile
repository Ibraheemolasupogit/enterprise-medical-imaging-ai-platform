.PHONY: install format format-check lint type-check test security validate-config validate-docs generate-synthetic-data validate-dataset verify-synthetic-data generate-dicom-fixtures discover-dicom validate-dicom-fixtures verify-dicom-ingestion quality-check-dicom verify-dicom-quality preprocess-dicom validate-preprocessed-volume verify-preprocessing register-synthetic-pair validate-registration verify-registration generate-localisation-fixtures localise-synthetic-regions validate-localisation verify-localisation prepare-segmentation-data train-segmentation evaluate-segmentation verify-segmentation prepare-classification-data train-classification evaluate-classification verify-classification analyse-synthetic-longitudinal validate-longitudinal verify-longitudinal validate-api test-api verify-api serve-api validate-reviewer-ui test-reviewer-ui verify-reviewer-ui serve-reviewer-ui validate-containers lint-dockerfiles scan-secrets scan-dependencies build-images scan-images generate-sbom container-smoke build-release-evidence validate-release-evidence register-model list-models approve-model build-monitoring-baseline run-monitoring simulate-monitoring-drift build-audit-evidence validate-monitoring-evidence verify-monitoring validate-helm render-kubernetes validate-kubernetes-policy deploy-local-kubernetes kubernetes-smoke build-kubernetes-evidence validate-kubernetes-evidence clean-local-kubernetes verify-release quality clean

PYTHON ?= python3
SYNTHETIC_DATA_DIR ?= data/synthetic/generated
DICOM_FIXTURE_DIR ?= data/dicom/fixtures
DICOM_DEID_DIR ?= data/dicom/deidentified
DICOM_AUDIT_PATH ?= data/dicom/audit/audit.json
DICOM_QUALITY_DIR ?= data/dicom/quality
PREPROCESSING_DIR ?= data/processed/preprocessing
REGISTRATION_FIXTURE_DIR ?= data/processed/registration-fixtures
REGISTRATION_DIR ?= data/processed/registration
LOCALISATION_FIXTURE_DIR ?= data/processed/localisation-fixtures
LOCALISATION_DIR ?= data/processed/localisation
SEGMENTATION_DATASET_DIR ?= ml/datasets/segmentation
SEGMENTATION_EXPERIMENT_DIR ?= ml/experiments/segmentation
SEGMENTATION_INFERENCE_DIR ?= ml/experiments/segmentation-inference
SEGMENTATION_ENV ?= MPLCONFIGDIR=/tmp/medical-imaging-platform-mplconfig
CLASSIFICATION_DATASET_DIR ?= ml/datasets/classification
CLASSIFICATION_EXPERIMENT_DIR ?= ml/experiments/classification
CLASSIFICATION_INFERENCE_DIR ?= ml/experiments/classification-inference
CLASSIFICATION_SYNTHETIC_CASES ?= 12
LONGITUDINAL_EXPERIMENT_DIR ?= ml/experiments/longitudinal
API_CONFIG ?= config/api.yaml
REVIEWER_UI_CONFIG ?= config/reviewer_ui.yaml
CONTAINER_CONFIG ?= config/container.yaml
RELEASE_EVIDENCE_DIR ?=

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

preprocess-dicom: verify-dicom-quality
	$(PYTHON) -m medical_imaging_platform preprocess-dicom $(DICOM_DEID_DIR) --study-uid auto --series-uid auto --output-dir $(PREPROCESSING_DIR) --overwrite

validate-preprocessed-volume:
	$(PYTHON) -m medical_imaging_platform validate-preprocessed-volume $$(find $(PREPROCESSING_DIR) -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)

verify-preprocessing: preprocess-dicom validate-preprocessed-volume
	$(PYTHON) -m medical_imaging_platform inspect-preprocessed-volume $$(find $(PREPROCESSING_DIR) -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)

register-synthetic-pair: generate-synthetic-data
	$(PYTHON) -m medical_imaging_platform generate-registration-fixtures --output-dir $(REGISTRATION_FIXTURE_DIR) --overwrite
	$(PYTHON) -m medical_imaging_platform register-volumes --fixed $(REGISTRATION_FIXTURE_DIR)/fixed --moving $(REGISTRATION_FIXTURE_DIR)/moving --fixed-temporal-label reference --moving-temporal-label followup --mode centre_of_mass --output-dir $(REGISTRATION_DIR) --overwrite
	$(PYTHON) -m medical_imaging_platform register-volumes --fixed $(REGISTRATION_FIXTURE_DIR)/fixed --moving $(REGISTRATION_FIXTURE_DIR)/moving --fixed-temporal-label reference --moving-temporal-label followup --mode rigid --output-dir $(REGISTRATION_DIR) --overwrite
	$(PYTHON) -m medical_imaging_platform register-volumes --fixed $(REGISTRATION_FIXTURE_DIR)/fixed --moving $(REGISTRATION_FIXTURE_DIR)/moving --fixed-temporal-label reference --moving-temporal-label followup --mode rigid_then_affine --output-dir $(REGISTRATION_DIR) --overwrite

validate-registration:
	$(PYTHON) -m medical_imaging_platform validate-registration $$(find $(REGISTRATION_DIR) -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)

verify-registration: register-synthetic-pair validate-registration
	$(PYTHON) -m medical_imaging_platform inspect-registration $$(find $(REGISTRATION_DIR) -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)

generate-localisation-fixtures:
	$(PYTHON) -m medical_imaging_platform generate-localisation-fixtures --output-dir $(LOCALISATION_FIXTURE_DIR) --overwrite

localise-synthetic-regions: generate-localisation-fixtures
	$(PYTHON) -m medical_imaging_platform localise-adrenal-regions $(LOCALISATION_FIXTURE_DIR) --left-mask $(LOCALISATION_FIXTURE_DIR)/left_adrenal_mask.npy --right-mask $(LOCALISATION_FIXTURE_DIR)/right_adrenal_mask.npy --output-dir $(LOCALISATION_DIR) --overwrite

validate-localisation:
	$(PYTHON) -m medical_imaging_platform validate-localisation $$(find $(LOCALISATION_DIR) -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)

verify-localisation: localise-synthetic-regions validate-localisation
	$(PYTHON) -m medical_imaging_platform inspect-localisation $$(find $(LOCALISATION_DIR) -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)

prepare-segmentation-data: generate-synthetic-data
	$(SEGMENTATION_ENV) $(PYTHON) -m medical_imaging_platform prepare-segmentation-data --synthetic-dataset-dir $(SYNTHETIC_DATA_DIR) --output-dir $(SEGMENTATION_DATASET_DIR) --overwrite

train-segmentation: prepare-segmentation-data
	$(SEGMENTATION_ENV) $(PYTHON) -m medical_imaging_platform train-segmentation $$(find $(SEGMENTATION_DATASET_DIR) -mindepth 1 -maxdepth 1 -type d | sort | head -n 1) --output-dir $(SEGMENTATION_EXPERIMENT_DIR) --overwrite

evaluate-segmentation:
	$(SEGMENTATION_ENV) $(PYTHON) -m medical_imaging_platform validate-segmentation-experiment $$(find $(SEGMENTATION_EXPERIMENT_DIR) -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)
	$(SEGMENTATION_ENV) $(PYTHON) -m medical_imaging_platform inspect-segmentation-experiment $$(find $(SEGMENTATION_EXPERIMENT_DIR) -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)

verify-segmentation: train-segmentation evaluate-segmentation
	$(SEGMENTATION_ENV) $(PYTHON) -m medical_imaging_platform segment-volume $$(find $(SEGMENTATION_DATASET_DIR) -path "*/synthetic-case-0001-current/image.npy" | sort | head -n 1) --checkpoint $$(find $(SEGMENTATION_EXPERIMENT_DIR) -mindepth 2 -maxdepth 2 -name best_model.pt | sort | head -n 1) --output-dir $(SEGMENTATION_INFERENCE_DIR)/positive --overwrite
	$(SEGMENTATION_ENV) $(PYTHON) -m medical_imaging_platform segment-volume $$(find $(SEGMENTATION_DATASET_DIR) -path "*/synthetic-case-0004-previous/image.npy" | sort | head -n 1) --checkpoint $$(find $(SEGMENTATION_EXPERIMENT_DIR) -mindepth 2 -maxdepth 2 -name best_model.pt | sort | head -n 1) --output-dir $(SEGMENTATION_INFERENCE_DIR)/negative --overwrite

prepare-classification-data:
	$(PYTHON) -m medical_imaging_platform generate-synthetic-data --output-dir $(SYNTHETIC_DATA_DIR) --cases $(CLASSIFICATION_SYNTHETIC_CASES) --overwrite
	$(SEGMENTATION_ENV) $(PYTHON) -m medical_imaging_platform prepare-classification-data --synthetic-dataset-dir $(SYNTHETIC_DATA_DIR) --output-dir $(CLASSIFICATION_DATASET_DIR) --overwrite

train-classification: prepare-classification-data
	$(SEGMENTATION_ENV) $(PYTHON) -m medical_imaging_platform train-classification $$(find $(CLASSIFICATION_DATASET_DIR) -mindepth 1 -maxdepth 1 -type d | sort | head -n 1) --output-dir $(CLASSIFICATION_EXPERIMENT_DIR) --overwrite

evaluate-classification:
	$(SEGMENTATION_ENV) $(PYTHON) -m medical_imaging_platform validate-classification-experiment $$(find $(CLASSIFICATION_EXPERIMENT_DIR) -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)
	$(SEGMENTATION_ENV) $(PYTHON) -m medical_imaging_platform inspect-classification-experiment $$(find $(CLASSIFICATION_EXPERIMENT_DIR) -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)

verify-classification: train-classification evaluate-classification
	$(SEGMENTATION_ENV) $(PYTHON) -m medical_imaging_platform classify-volume $$(find $(CLASSIFICATION_DATASET_DIR) -path "*/synthetic-case-0001-current-left/image.npy" | sort | head -n 1) --checkpoint $$(find $(CLASSIFICATION_EXPERIMENT_DIR) -mindepth 2 -maxdepth 2 -name best_model.pt | sort | head -n 1) --calibration $$(find $(CLASSIFICATION_EXPERIMENT_DIR) -mindepth 2 -maxdepth 2 -name calibration.json | sort | head -n 1) --threshold-policy $$(find $(CLASSIFICATION_EXPERIMENT_DIR) -mindepth 2 -maxdepth 2 -name threshold_policy.json | sort | head -n 1) --output-dir $(CLASSIFICATION_INFERENCE_DIR)/positive --overwrite
	$(SEGMENTATION_ENV) $(PYTHON) -m medical_imaging_platform classify-volume $$(find $(CLASSIFICATION_DATASET_DIR) -path "*/synthetic-case-0004-previous-right/image.npy" | sort | head -n 1) --checkpoint $$(find $(CLASSIFICATION_EXPERIMENT_DIR) -mindepth 2 -maxdepth 2 -name best_model.pt | sort | head -n 1) --calibration $$(find $(CLASSIFICATION_EXPERIMENT_DIR) -mindepth 2 -maxdepth 2 -name calibration.json | sort | head -n 1) --threshold-policy $$(find $(CLASSIFICATION_EXPERIMENT_DIR) -mindepth 2 -maxdepth 2 -name threshold_policy.json | sort | head -n 1) --output-dir $(CLASSIFICATION_INFERENCE_DIR)/negative --overwrite

analyse-synthetic-longitudinal:
	$(PYTHON) -m medical_imaging_platform generate-synthetic-data --output-dir $(SYNTHETIC_DATA_DIR) --overwrite
	$(PYTHON) -m medical_imaging_platform analyse-longitudinal-pair --previous-mask $(SYNTHETIC_DATA_DIR)/synthetic-case-0001/previous_lesion_mask.npy --current-mask $(SYNTHETIC_DATA_DIR)/synthetic-case-0001/current_lesion_mask.npy --previous-spacing 2.5 2.5 2.5 --current-spacing 2.5 2.5 2.5 --case-id synthetic-case-0001 --research-subject-id research-subject-0001 --side left --previous-timepoint previous --current-timepoint current --registration-run-id synthetic-registration-pass --segmentation-run-id synthetic-segmentation-pass --classification-run-id synthetic-classification-pass --output-dir $(LONGITUDINAL_EXPERIMENT_DIR) --overwrite
	$(PYTHON) -m medical_imaging_platform analyse-longitudinal-pair --previous-mask $(SYNTHETIC_DATA_DIR)/synthetic-case-0002/previous_lesion_mask.npy --current-mask $(SYNTHETIC_DATA_DIR)/synthetic-case-0002/current_lesion_mask.npy --previous-spacing 2.5 2.5 2.5 --current-spacing 2.5 2.5 2.5 --case-id synthetic-case-0002 --research-subject-id research-subject-0002 --side right --previous-timepoint previous --current-timepoint current --registration-run-id synthetic-registration-pass --segmentation-run-id synthetic-segmentation-pass --classification-run-id synthetic-classification-pass --output-dir $(LONGITUDINAL_EXPERIMENT_DIR) --overwrite
	$(PYTHON) -m medical_imaging_platform analyse-longitudinal-pair --previous-mask $(SYNTHETIC_DATA_DIR)/synthetic-case-0003/previous_lesion_mask.npy --current-mask $(SYNTHETIC_DATA_DIR)/synthetic-case-0003/current_lesion_mask.npy --previous-spacing 2.5 2.5 2.5 --current-spacing 2.5 2.5 2.5 --case-id synthetic-case-0003 --research-subject-id research-subject-0003 --side left --previous-timepoint previous --current-timepoint current --registration-run-id synthetic-registration-pass --segmentation-run-id synthetic-segmentation-pass --classification-run-id synthetic-classification-pass --output-dir $(LONGITUDINAL_EXPERIMENT_DIR) --overwrite
	$(PYTHON) -m medical_imaging_platform analyse-longitudinal-pair --previous-mask $(SYNTHETIC_DATA_DIR)/synthetic-case-0004/previous_lesion_mask.npy --current-mask $(SYNTHETIC_DATA_DIR)/synthetic-case-0004/current_lesion_mask.npy --previous-spacing 2.5 2.5 2.5 --current-spacing 2.5 2.5 2.5 --case-id synthetic-case-0004 --research-subject-id research-subject-0004 --side right --previous-timepoint previous --current-timepoint current --registration-run-id synthetic-registration-pass --segmentation-run-id synthetic-segmentation-pass --classification-run-id synthetic-classification-pass --output-dir $(LONGITUDINAL_EXPERIMENT_DIR) --overwrite
	$(PYTHON) -m medical_imaging_platform analyse-longitudinal-pair --previous-mask $(SYNTHETIC_DATA_DIR)/synthetic-case-0005/previous_lesion_mask.npy --current-mask $(SYNTHETIC_DATA_DIR)/synthetic-case-0005/current_lesion_mask.npy --previous-spacing 2.5 2.5 2.5 --current-spacing 2.5 2.5 2.5 --case-id synthetic-case-0005 --research-subject-id research-subject-0005 --side left --previous-timepoint previous --current-timepoint current --registration-run-id synthetic-registration-pass --segmentation-run-id synthetic-segmentation-pass --classification-run-id synthetic-classification-pass --output-dir $(LONGITUDINAL_EXPERIMENT_DIR) --overwrite
	-$(PYTHON) -m medical_imaging_platform analyse-longitudinal-pair --previous-mask $(SYNTHETIC_DATA_DIR)/synthetic-case-0001/previous_lesion_mask.npy --current-mask $(SYNTHETIC_DATA_DIR)/synthetic-case-0001/current_lesion_mask.npy --previous-spacing 2.5 2.5 2.5 --current-spacing 2.5 2.5 2.5 --case-id synthetic-case-0001-failed-quality --research-subject-id research-subject-0001-failed --side left --previous-timepoint previous --current-timepoint current --registration-run-id synthetic-registration-fail --registration-status FAIL --segmentation-run-id synthetic-segmentation-pass --classification-run-id synthetic-classification-abstained --classification-abstention-status ABSTAINED --output-dir $(LONGITUDINAL_EXPERIMENT_DIR) --overwrite

validate-longitudinal:
	$(PYTHON) -m medical_imaging_platform validate-longitudinal-analysis $(LONGITUDINAL_EXPERIMENT_DIR)/longitudinal-synthetic-case-0001-left-m10-longitudinal-v1
	$(PYTHON) -m medical_imaging_platform validate-longitudinal-analysis $(LONGITUDINAL_EXPERIMENT_DIR)/longitudinal-synthetic-case-0002-right-m10-longitudinal-v1
	$(PYTHON) -m medical_imaging_platform validate-longitudinal-analysis $(LONGITUDINAL_EXPERIMENT_DIR)/longitudinal-synthetic-case-0003-left-m10-longitudinal-v1
	$(PYTHON) -m medical_imaging_platform validate-longitudinal-analysis $(LONGITUDINAL_EXPERIMENT_DIR)/longitudinal-synthetic-case-0004-right-m10-longitudinal-v1
	$(PYTHON) -m medical_imaging_platform validate-longitudinal-analysis $(LONGITUDINAL_EXPERIMENT_DIR)/longitudinal-synthetic-case-0005-left-m10-longitudinal-v1
	$(PYTHON) -m medical_imaging_platform validate-longitudinal-analysis $(LONGITUDINAL_EXPERIMENT_DIR)/longitudinal-synthetic-case-0001-failed-quality-left-m10-longitudinal-v1
	$(PYTHON) -m medical_imaging_platform inspect-longitudinal-analysis $(LONGITUDINAL_EXPERIMENT_DIR)/longitudinal-synthetic-case-0002-right-m10-longitudinal-v1
	git check-ignore -q $(LONGITUDINAL_EXPERIMENT_DIR)/longitudinal-synthetic-case-0001-left-m10-longitudinal-v1/analysis_manifest.json

verify-longitudinal: analyse-synthetic-longitudinal validate-longitudinal

validate-api:
	$(PYTHON) -m medical_imaging_platform validate-api-config --config $(API_CONFIG)

test-api:
	$(PYTHON) -m pytest tests/test_api.py --no-cov

verify-api: test-api
	git check-ignore -q ml/experiments/api

serve-api:
	$(PYTHON) -m medical_imaging_platform serve-api --config $(API_CONFIG)

validate-reviewer-ui:
	$(PYTHON) -m medical_imaging_platform validate-reviewer-ui-config --config $(REVIEWER_UI_CONFIG)

test-reviewer-ui:
	$(PYTHON) -m pytest tests/test_reviewer_ui.py --no-cov

verify-reviewer-ui: validate-reviewer-ui test-reviewer-ui
	git check-ignore -q reports/generated/reviewer-sessions/example-review/review_decision.json

serve-reviewer-ui:
	$(PYTHON) -m medical_imaging_platform serve-reviewer-ui --config $(REVIEWER_UI_CONFIG)

validate-containers:
	$(PYTHON) -m medical_imaging_platform validate-container-config --config $(CONTAINER_CONFIG)
	$(PYTHON) -m medical_imaging_platform validate-api-config --config config/container/api.yaml
	$(PYTHON) -m medical_imaging_platform validate-reviewer-ui-config --config config/container/reviewer_ui.yaml

lint-dockerfiles:
	$(PYTHON) -m medical_imaging_platform inspect-container-security --config $(CONTAINER_CONFIG)

scan-secrets:
	$(PYTHON) -m medical_imaging_platform scan-release-secrets --config $(CONTAINER_CONFIG)

scan-dependencies:
	$(PYTHON) -m medical_imaging_platform scan-release-dependencies --config $(CONTAINER_CONFIG)

build-images:
	docker compose build --pull=false

scan-images:
	$(PYTHON) -m medical_imaging_platform scan-release-images --config $(CONTAINER_CONFIG)

generate-sbom:
	$(PYTHON) -m medical_imaging_platform generate-release-sbom --config $(CONTAINER_CONFIG)

container-smoke:
	$(PYTHON) -m medical_imaging_platform run-container-smoke-tests --config $(CONTAINER_CONFIG)

build-release-evidence:
	$(PYTHON) -m medical_imaging_platform build-release-manifest --config $(CONTAINER_CONFIG) --overwrite

validate-release-evidence:
	$(PYTHON) -m medical_imaging_platform validate-release-evidence $(if $(RELEASE_EVIDENCE_DIR),--release-dir $(RELEASE_EVIDENCE_DIR),)

register-model:
	$(PYTHON) -m medical_imaging_platform register-model

list-models:
	$(PYTHON) -m medical_imaging_platform list-models

approve-model: register-model
	$(PYTHON) -m medical_imaging_platform approve-model --model-name synthetic-segmentation-baseline --version m14-segmentation-synthetic-v1 --approved-by m14-governance-reviewer --approval-ticket M14-SYNTHETIC-SEG-APPROVAL --rationale "Synthetic engineering baseline approved for local monitoring evidence only."
	$(PYTHON) -m medical_imaging_platform approve-model --model-name synthetic-classification-baseline --version m14-classification-synthetic-v1 --approved-by m14-governance-reviewer --approval-ticket M14-SYNTHETIC-CLS-APPROVAL --rationale "Synthetic engineering baseline approved for local monitoring evidence only."

build-monitoring-baseline: approve-model
	$(PYTHON) -m medical_imaging_platform build-monitoring-baseline

run-monitoring:
	$(PYTHON) -m medical_imaging_platform run-monitoring

simulate-monitoring-drift:
	$(PYTHON) -m medical_imaging_platform simulate-monitoring-drift

build-audit-evidence:
	$(PYTHON) -m medical_imaging_platform build-audit-evidence

validate-monitoring-evidence:
	$(PYTHON) -m medical_imaging_platform validate-monitoring-evidence

verify-monitoring: build-monitoring-baseline run-monitoring simulate-monitoring-drift build-audit-evidence validate-monitoring-evidence
	git check-ignore -q reports/generated/monitoring/monitoring_baseline.json
	git check-ignore -q reports/generated/registry/registry_manifest.json
	git check-ignore -q reports/generated/audit/audit_log.jsonl

validate-helm:
	$(PYTHON) -m medical_imaging_platform validate-helm

render-kubernetes:
	$(PYTHON) -m medical_imaging_platform render-kubernetes

validate-kubernetes-policy: render-kubernetes
	$(PYTHON) -m medical_imaging_platform validate-kubernetes-policy

deploy-local-kubernetes:
	$(PYTHON) -m medical_imaging_platform deploy-local-kubernetes

kubernetes-smoke:
	$(PYTHON) -m medical_imaging_platform kubernetes-smoke

build-kubernetes-evidence: render-kubernetes validate-kubernetes-policy
	$(PYTHON) -m medical_imaging_platform build-kubernetes-evidence

validate-kubernetes-evidence:
	$(PYTHON) -m medical_imaging_platform validate-kubernetes-evidence

clean-local-kubernetes:
	$(PYTHON) -m medical_imaging_platform clean-local-kubernetes

verify-release: quality validate-containers lint-dockerfiles scan-secrets scan-dependencies build-images generate-sbom scan-images container-smoke build-release-evidence validate-release-evidence
	git check-ignore -q reports/generated/releases/example/release_manifest.json

quality: format-check lint type-check validate-config validate-docs test security

clean:
	rm -rf .coverage coverage.xml .mypy_cache .pytest_cache .ruff_cache src/medical_imaging_platform/__pycache__ src/medical_imaging_platform/deidentification/__pycache__ src/medical_imaging_platform/ingestion/__pycache__ src/medical_imaging_platform/localisation/__pycache__ src/medical_imaging_platform/preprocessing/__pycache__ src/medical_imaging_platform/quality_control/__pycache__ src/medical_imaging_platform/registration/__pycache__ src/medical_imaging_platform/segmentation/__pycache__ src/medical_imaging_platform/synthetic/__pycache__ src/medical_imaging_platform/utils/__pycache__ tests/__pycache__
