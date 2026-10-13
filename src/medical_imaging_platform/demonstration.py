"""Bounded CPU demonstration of existing platform workflows; no external assets."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from fastapi.testclient import TestClient
from pydicom.uid import UID

from medical_imaging_platform.api.app import create_app
from medical_imaging_platform.classification.benchmark import run_classification_benchmark
from medical_imaging_platform.classification.dataset import prepare_classification_dataset
from medical_imaging_platform.detection.benchmark import run_detection_benchmark
from medical_imaging_platform.detection.dataset import prepare_detection_dataset
from medical_imaging_platform.embeddings.benchmark import run_benchmark
from medical_imaging_platform.embeddings.config import load_embedding_config
from medical_imaging_platform.embeddings.core import EmbeddingIndex, query_index
from medical_imaging_platform.embeddings.extractor import generate_embeddings
from medical_imaging_platform.embeddings.metrics import evaluate_index
from medical_imaging_platform.evidence import (
    ArtifactBinding,
    EvidenceEnvelope,
    evidence_checksum,
    sha256_file,
    validate_envelope,
    write_envelope,
)
from medical_imaging_platform.ingestion.fixtures import build_fixture_dataset
from medical_imaging_platform.localisation.pipeline import localise_adrenal_regions
from medical_imaging_platform.longitudinal.pipeline import analyse_longitudinal_pair
from medical_imaging_platform.preprocessing.pipeline import preprocess_dicom_series
from medical_imaging_platform.preprocessing_baseline.runner import run_preprocessing_baseline
from medical_imaging_platform.reviewer_ui.export import export_review_session
from medical_imaging_platform.reviewer_ui.review import create_review_decision
from medical_imaging_platform.segmentation.dataset import prepare_segmentation_dataset
from medical_imaging_platform.segmentation.inference import segment_volume
from medical_imaging_platform.segmentation.trainer import train_segmentation_experiment
from medical_imaging_platform.synthetic.generator import load_synthetic_config
from medical_imaging_platform.synthetic.io import generate_dataset
from medical_imaging_platform.synthetic.manifest import MANIFEST_FILENAME
from medical_imaging_platform.utils.config import (
    load_api_config,
    load_classification_config,
    load_detection_config,
    load_dicom_ingestion_config,
    load_localisation_config,
    load_longitudinal_config,
    load_preprocessing_baseline_config,
    load_preprocessing_config,
    load_quality_control_config,
    load_reviewer_ui_config,
    load_segmentation_config,
)


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config(path: Path, section: str, config: Any) -> None:
    path.write_text(
        yaml.safe_dump({"settings": {section: config.model_dump(mode="json")}}), encoding="utf-8"
    )


def validate_demo(root: Path) -> dict[str, Any]:
    """Verify index metadata, bounded references, sidecars and all generated file hashes."""
    root = root.resolve()
    payload = json.loads((root / "evidence_index.json").read_text(encoding="utf-8"))
    if payload.get("evidence_checksum") != evidence_checksum(payload):
        raise ValueError("Demo evidence index checksum mismatch")
    for name, digest in payload["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != digest:
            raise ValueError("Demo evidence file checksum mismatch")
    for entry in payload["artifacts"]:
        for reference in (entry["envelope"], entry["artifact"]):
            if not (root / reference).resolve().is_relative_to(root):
                raise ValueError("Demo artifact reference escapes output root")
        envelope = validate_envelope(root / entry["envelope"], root / entry["artifact"])
        if envelope.artifact_reference != entry["artifact"]:
            raise ValueError("Demo artifact reference mismatch")
        if envelope.run_id != entry["run_id"]:
            raise ValueError("Demo run reference mismatch")
        if envelope.source_id is not None:
            source = (root / envelope.source_id).resolve()
            if not source.is_relative_to(root) or sha256_file(source) != envelope.source_checksum:
                raise ValueError("Demo source checksum mismatch")
    return cast(dict[str, Any], payload)


def run_platform_demo(output_root: Path) -> dict[str, Any]:
    """Train tiny existing models and produce honest, cross-referenced evidence."""
    root = output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Platform demo requires a new or empty output directory.")
    root.mkdir(parents=True, exist_ok=True)
    config_dir = root / "config"
    config_dir.mkdir()
    envelopes = root / "envelopes"
    envelopes.mkdir()
    artifacts: list[dict[str, str]] = []

    def record(
        kind: str,
        run_id: str,
        artifact: Path,
        *,
        source: Path | None = None,
        role: str = "derived",
        status: str = "PASS",
        model: str | None = None,
        checkpoint: Path | None = None,
        config: Path | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        envelope = EvidenceEnvelope.model_validate(
            {
                "artifact_type": kind,
                "run_id": run_id,
                "artifact_reference": str(artifact.relative_to(root)),
                "artifact_checksum": sha256_file(artifact),
                "source_id": str(source.relative_to(root)) if source else None,
                "source_checksum": sha256_file(source) if source else None,
                "model_family": model,
                "checkpoint_checksum": sha256_file(checkpoint) if checkpoint else None,
                "config_checksum": sha256_file(config) if config else None,
                "quality_status": status,
                "data_role": role,
                "warnings": warnings or [],
            }
        )
        sidecar = envelopes / f"{len(artifacts):02d}-{kind}.json"
        write_envelope(sidecar, envelope)
        artifacts.append(
            {
                "run_id": run_id,
                "artifact": envelope.artifact_reference,
                "envelope": str(sidecar.relative_to(root)),
            }
        )

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        synthetic_root = root / "synthetic"
        data = load_synthetic_config(Path("config/data.yaml"))
        dataset = generate_dataset(
            data, synthetic_root, case_count=12, random_seed=data.random_seed
        )
        source_manifest = synthetic_root / MANIFEST_FILENAME
        record("synthetic_dataset", dataset.dataset_id, source_manifest, role="reference")

        # DICOM is a separate small engineering fixture, not a conversion of these model volumes.
        dicom_root = root / "dicom"
        dicom_root.mkdir()
        for index in range(4):
            fixture = build_fixture_dataset(
                index=index,
                study_uid="1.2.826.0.1.3680043.10.54321.23.1",
                series_uid="1.2.826.0.1.3680043.10.54321.23.2",
                sop_uid=f"1.2.826.0.1.3680043.10.54321.23.3.{index + 1}",
            )
            fixture.file_meta.ImplementationClassUID = UID("1.2.826.0.1.3680043.10.54321.23.9")
            fixture.save_as(dicom_root / f"slice-{index:03d}.dcm", enforce_file_format=True)
        ingestion = load_dicom_ingestion_config(Path("config/data.yaml"))
        pre = preprocess_dicom_series(
            dicom_root,
            output_root=root / "preprocessing",
            preprocessing_config=load_preprocessing_config(Path("config/preprocessing.yaml")),
            qc_config=load_quality_control_config(Path("config/data.yaml")),
            dicom_max_files=ingestion.max_files,
            max_file_size_bytes=ingestion.max_file_size_bytes,
        )
        pre_dir = Path(pre.output_paths.output_dir)
        record(
            "preprocessing",
            pre.run_id,
            pre_dir / "metadata.json",
            source=dicom_root / "slice-000.dcm",
            warnings=pre.warnings,
        )
        baseline = run_preprocessing_baseline(
            [pre_dir],
            output_root=root / "baseline",
            config=load_preprocessing_baseline_config(Path("config/preprocessing_baseline.yaml")),
        )
        baseline_path = Path(baseline.output_paths.summary)
        record(
            "preprocessing_baseline",
            baseline.baseline_run_id,
            baseline_path,
            source=pre_dir / "volume.npy",
            warnings=baseline.warnings,
        )
        loc = localise_adrenal_regions(
            pre_dir,
            output_root=root / "localisation",
            config=load_localisation_config(Path("config/localisation.yaml")),
        )
        loc_path = Path(loc.output_paths.localisation_json)
        record(
            "localisation",
            loc.localisation_run_id,
            loc_path,
            source=pre_dir / "metadata.json",
            status=loc.overall_status,
            warnings=["Atlas placeholder; independent model datasets do not consume these ROIs."],
        )

        seg_config = load_segmentation_config(Path("config/segmentation.yaml")).model_copy(
            update={"epochs": 1}
        )
        seg_path = config_dir / "segmentation.yaml"
        _config(seg_path, "segmentation", seg_config)
        seg_data = prepare_segmentation_dataset(
            synthetic_root,
            output_root=root / "segmentation-data",
            config=seg_config,
            overwrite=False,
        )
        seg_dir = root / "segmentation-data" / seg_data.dataset_id
        seg = train_segmentation_experiment(
            seg_dir, output_root=root / "segmentation", config=seg_config, overwrite=False
        )
        seg_exp = root / "segmentation" / seg["experiment_id"]
        seg_checkpoint = seg_exp / seg["paths"]["best_model"]
        record(
            "segmentation",
            seg["experiment_id"],
            seg_exp / "model_manifest.json",
            source=source_manifest,
            model="monai_3d_unet",
            checkpoint=seg_checkpoint,
            config=seg_path,
            status=seg["status"],
        )
        case_id = next(sample.case_id for sample in seg_data.samples if sample.split == "test")
        pair = [
            next(s for s in seg_data.samples if s.case_id == case_id and s.timepoint == t)
            for t in ("previous", "current")
        ]
        masks = []
        for sample in pair:
            prediction_dir = root / "predictions" / sample.sample_id
            prediction = segment_volume(
                seg_dir / sample.image,
                checkpoint_path=seg_checkpoint,
                output_dir=prediction_dir,
                config=seg_config,
                overwrite=False,
            )
            mask = prediction_dir / "predicted_mask.npy"
            masks.append(mask)
            record(
                "segmentation_prediction",
                sample.sample_id,
                mask,
                source=seg_dir / sample.image,
                role="predicted",
                model="monai_3d_unet",
                checkpoint=seg_checkpoint,
                config=seg_path,
                status="PASS_WITH_WARNINGS" if prediction["warnings"] else "PASS",
                warnings=prediction["warnings"],
            )
        longitudinal = analyse_longitudinal_pair(
            previous_mask_path=masks[0],
            current_mask_path=masks[1],
            previous_spacing_mm=pair[0].spacing_mm,
            current_spacing_mm=pair[1].spacing_mm,
            case_id=case_id,
            research_subject_id=pair[0].research_subject_id,
            side=next(
                record.lesion_side for record in dataset.records if record.case_id == case_id
            ),
            previous_timepoint="previous",
            current_timepoint="current",
            output_root=root / "longitudinal",
            config=load_longitudinal_config(Path("config/longitudinal.yaml")),
            segmentation_run_ids=[seg["experiment_id"], *(s.sample_id for s in pair)],
            upstream_quality_statuses={"segmentation": seg["status"]},
        )
        long_path = root / "longitudinal" / longitudinal["analysis_id"] / "pair_manifest.json"
        record(
            "longitudinal",
            longitudinal["analysis_id"],
            long_path,
            source=masks[0],
            role="predicted",
            status=longitudinal["status"],
            warnings=[
                (
                    "Consumes predicted masks; empty predictions are possible. No clinical "
                    "change assessment."
                )
            ],
        )

        cls_config = load_classification_config(Path("config/classification.yaml")).model_copy(
            update={"epochs": 1}
        )
        cls_data = prepare_classification_dataset(
            synthetic_root,
            output_root=root / "classification-data",
            config=cls_config,
            overwrite=False,
        )
        cls_dir = root / "classification-data" / cls_data.dataset_id
        cls = run_classification_benchmark(
            cls_dir, output_root=root / "classification", config=cls_config, overwrite=False
        )
        cls_bench = root / "classification" / cls["benchmark_id"]
        record(
            "classification_benchmark",
            cls["benchmark_id"],
            cls_bench / "classification_benchmark_comparison.json",
            source=cls_dir / "classification_manifest.json",
            role="derived",
        )
        policy = load_embedding_config(Path("config/embeddings.yaml"))
        indexes = []
        class_paths = {}
        for family, model in cls["models"].items():
            path = config_dir / f"{family}.yaml"
            _config(path, "classification", cls_config.model_copy(update={"model_family": family}))
            experiment = cls_bench / model["experiment_dir"]
            checkpoint = experiment / "best_model.pt"
            class_paths[family] = (path, checkpoint, experiment)
            directory = root / "embeddings" / family
            index_manifest = generate_embeddings(
                cls_dir, path, checkpoint, family, directory, policy=policy
            )
            embedding_index = EmbeddingIndex.load(directory)
            query_id = next(
                r["source_sample_id"]
                for r in embedding_index.records
                if r["split"] in policy.query_splits
            )
            _json(
                directory / "query.json",
                {
                    "query_sample_id": query_id,
                    "results": query_index(embedding_index, query_id, 5, policy),
                },
            )
            _json(directory / "evaluation.json", evaluate_index(embedding_index, policy))
            record(
                "embedding_index",
                str(index_manifest["index_run_id"]),
                directory / "embedding_matrix.npy",
                source=cls_dir / "classification_manifest.json",
                model=family,
                checkpoint=checkpoint,
                config=path,
            )
            indexes.append(directory)
        retrieval = run_benchmark(indexes[0], indexes[1], root / "retrieval", policy=policy)
        record(
            "retrieval_benchmark",
            "cnn-vit-retrieval",
            root / "retrieval" / "embedding_benchmark_comparison.json",
            source=indexes[0] / "embedding_manifest.json",
        )

        det_config = load_detection_config(Path("config/detection.yaml"))
        det_path = config_dir / "detection.yaml"
        _config(det_path, "detection", det_config)
        det_data = prepare_detection_dataset(
            synthetic_root, output_root=root / "detection-data", config=det_config, overwrite=False
        )
        det_dir = root / "detection-data" / det_data.dataset_id
        detection = run_detection_benchmark(
            det_dir, output_root=root / "detection", config=det_config, overwrite=False
        )
        det_bench = root / "detection" / detection["benchmark_id"]
        record(
            "detection_benchmark",
            detection["benchmark_id"],
            det_bench / "detection_benchmark_comparison.json",
            source=det_dir / "detection_dataset_manifest.json",
            role="derived",
            config=det_path,
        )

        cnn_config, cnn_checkpoint, cnn_experiment = class_paths["small_3d_cnn"]
        api_config = load_api_config(Path("config/api.yaml")).model_copy(
            update={
                "allowed_input_roots": [root],
                "allowed_evidence_roots": [root],
                "output_directory": root / "api-outputs",
                "segmentation_checkpoint": seg_checkpoint,
                "segmentation_config": seg_path,
                "classification_checkpoint": cnn_checkpoint,
                "classification_config": cnn_config,
                "classification_calibration": cnn_experiment / "calibration.json",
                "classification_threshold_policy": cnn_experiment / "threshold_policy.json",
                "detection_checkpoint": det_bench
                / detection["models"]["faster_rcnn"]["experiment_dir"]
                / "best_model.pt",
                "detection_config": det_path,
                "embedding_indexes": indexes,
            }
        )
        api_path = config_dir / "api.yaml"
        _config(api_path, "api", api_config)
        ui_config = load_reviewer_ui_config(Path("config/reviewer_ui.yaml")).model_copy(
            update={"review_output_directory": root / "reviews"}
        )
        _config(config_dir / "reviewer_ui.yaml", "reviewer_ui", ui_config)
        with TestClient(create_app(api_path)) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/ready").status_code == 200
            query = next(
                r["source_sample_id"]
                for r in EmbeddingIndex.load(indexes[0]).records
                if r["split"] == "test"
            )
            search = client.post(
                "/v1/embeddings/search",
                json={"index_directory": str(indexes[0]), "sample_id": query, "top_k": 5},
            )
            assert search.status_code == 200, search.text
            seg_response = client.post(
                "/v1/segmentation/predict",
                json={
                    "input_path": str(seg_dir / pair[0].image),
                    "spacing_mm": list(pair[0].spacing_mm),
                },
                headers={"X-Request-ID": "m23-segmentation-review"},
            )
            assert seg_response.status_code == 200, seg_response.text
        _json(root / "reviewer_search.json", search.json())
        _json(root / "reviewer_segmentation.json", seg_response.json())
        binding = ArtifactBinding.model_validate(seg_response.json()["artifact"])
        decision = create_review_decision(
            request_id="m23-segmentation-review",
            evidence_type="segmentation",
            evidence_id=pair[0].sample_id,
            model_engineering_label="synthetic-mask",
            quality_status=seg["status"],
            reviewer_decision="needs_secondary_review",
            artifact=binding,
            review_notes="Canonical engineering smoke; requires human visual review.",
        )
        export_review_session(
            decision=decision, evidence_summary=seg_response.json(), output_root=root / "reviews"
        )
        context = {
            "device": "cpu",
            "threads": 1,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "seed": data.random_seed,
            "model_seeds": {
                "segmentation": seg_config.random_seed,
                "classification": cls_config.random_seed,
                "detection": det_config.random_seed,
            },
            "cases": 12,
            "epochs": 1,
            "classification_samples": len(cls_data.samples),
            "detection_samples": len(det_data.samples),
            "segmentation_samples": len(seg_data.samples),
        }
        comparisons = {
            "context": context,
            "classification": cls["models"],
            "detection": detection["models"],
            "retrieval": retrieval["models"],
            "limitations": [
                "Synthetic one-epoch smoke only; not clinical performance.",
                "Latency and timestamps vary with execution; seeds, data and ordering are fixed.",
            ],
        }
        _json(root / "engineering_comparison.json", comparisons)
        (root / "engineering_comparison.md").write_text(
            "# Engineering comparison\n\n"
            "Measured CPU smoke evidence; no universal model ranking.\n\n```json\n"
            + json.dumps(comparisons, indent=2, sort_keys=True)
            + "\n```\n",
            encoding="utf-8",
        )
        summary = {
            "schema_version": "M23-v1",
            "status": "PASS",
            "run_id": "m23-platform-demo-seed-20260724",
            "context": context,
            "artifacts": artifacts,
            "workflow": {
                "dicom": (
                    "Separate synthetic DICOM → QC/preprocessing → classical baseline/localisation."
                ),
                "models": (
                    "Shared synthetic labelled volumes → independent supervised "
                    "segmentation/classification/detection datasets."
                ),
                "retrieval": (
                    "Trained CNN/ViT classification checkpoints → embeddings → exact retrieval."
                ),
                "longitudinal": (
                    "Actual segmentation predictions → longitudinal "
                    "measurements; reference masks are training/evaluation labels."
                ),
                "review": (
                    "Actual API responses and separate artifact-bound "
                    "engineering review; not human acceptance."
                ),
            },
            "capability_status": {
                "model_smoke": "executed/validated",
                "reviewer_controls": "implemented",
                "operations": "simulated",
                "kubernetes": "static design",
                "aws": "target state",
            },
            "files": {
                str(p.relative_to(root)): sha256_file(p)
                for p in sorted(root.rglob("*"))
                if p.is_file()
            },
        }
        summary["evidence_checksum"] = evidence_checksum(summary)
        _json(root / "evidence_index.json", summary)
        return validate_demo(root)
    finally:
        torch.set_num_threads(previous_threads)
