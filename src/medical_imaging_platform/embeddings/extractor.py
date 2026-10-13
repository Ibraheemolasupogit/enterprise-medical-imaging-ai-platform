from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from medical_imaging_platform.classification.checkpoint import load_state_dict
from medical_imaging_platform.classification.model_factory import (
    architecture_metadata,
    build_classifier,
)
from medical_imaging_platform.classification.models import ClassificationConfig
from medical_imaging_platform.embeddings.config import EmbeddingConfig, load_embedding_config
from medical_imaging_platform.embeddings.core import l2_normalize, sha256_file, sha256_json
from medical_imaging_platform.evidence import EvidenceEnvelope, write_envelope


def generate_embeddings(
    dataset_dir: Path,
    config_path: Path,
    checkpoint: Path,
    encoder_family: str,
    output_dir: Path,
    overwrite: bool = False,
    *,
    policy: EmbeddingConfig | None = None,
) -> dict[str, object]:
    policy = policy or load_embedding_config("config/embeddings.yaml")
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(str(output_dir))
    mp = dataset_dir / "classification_manifest.json"
    if not mp.exists():
        mp = dataset_dir / "classification_dataset_manifest.json"
    if not mp.exists():
        raise FileNotFoundError(str(mp))
    source = json.loads(mp.read_text())
    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    raw = raw.get("settings", {}).get("classification", raw)
    config = ClassificationConfig.model_validate(raw)
    if config.model_family != encoder_family:
        raise ValueError("encoder family/config mismatch")
    model = build_classifier(config)
    model.load_state_dict(load_state_dict(Path(checkpoint)))
    model.eval()
    vectors = []
    records = []
    start = time.perf_counter()
    with torch.no_grad():
        for sample in sorted(source["samples"], key=lambda x: x["sample_id"]):
            path = dataset_dir / sample["image"]
            image = np.load(path, allow_pickle=False).astype(np.float32)
            if not np.isfinite(image).all():
                raise ValueError("non-finite source image")
            tensor = torch.from_numpy(image)
            tensor = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
            vectors.append(model.extract_features(tensor.unsqueeze(0))[0].cpu().numpy())
            records.append(
                {
                    "embedding_id": f"{encoder_family}:{sample['sample_id']}",
                    "source_sample_id": sample["sample_id"],
                    "case_id": sample.get("case_id"),
                    "research_subject_id": sample.get("research_subject_id"),
                    "split": sample.get("split"),
                    "scenario": sample.get("scenario", sample.get("label_name")),
                    "label": sample.get("label"),
                    "source_checksum": sha256_file(path),
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = l2_normalize(np.asarray(vectors))
    matrix_path = output_dir / "embedding_matrix.npy"
    np.save(matrix_path, matrix, allow_pickle=False)
    manifest = {
        "policy_version": policy.policy_version,
        "retrieval_policy": policy.model_dump(mode="json"),
        "index_run_id": sha256_json([x["sample_id"] for x in source["samples"]])[:16],
        "dataset_id": source.get("dataset_id", dataset_dir.name),
        "source_dataset_manifest_checksum": sha256_file(mp),
        "encoder_family": encoder_family,
        "encoder_architecture": architecture_metadata(config),
        "encoder_checkpoint_checksum": sha256_file(Path(checkpoint)),
        "embedding_dimension": int(matrix.shape[1]),
        "normalization": "l2",
        "records": records,
        "matrix_checksum": sha256_file(matrix_path),
        "config_checksum": sha256_file(Path(config_path)),
        "gallery_splits": policy.gallery_splits,
        "query_splits": policy.query_splits,
        "generation_seconds": time.perf_counter() - start,
        "warnings": [
            "Synthetic/de-identified engineering retrieval only; not diagnostic similarity."
        ],
    }
    (output_dir / "embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    write_envelope(
        output_dir / "evidence_envelope.json",
        EvidenceEnvelope(
            artifact_type="embedding_index",
            run_id=str(manifest["index_run_id"]),
            artifact_reference="embedding_matrix.npy",
            artifact_checksum=sha256_file(matrix_path),
            source_id=str(manifest["dataset_id"]),
            source_checksum=sha256_file(mp),
            model_family=encoder_family,
            checkpoint_checksum=sha256_file(Path(checkpoint)),
            config_checksum=sha256_file(Path(config_path)),
            quality_status="PASS",
            warnings=["Embedding-space similarity is not diagnostic similarity."],
        ),
    )
    (output_dir / "evidence").mkdir(exist_ok=True)
    (output_dir / "evidence" / "checksum_manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    n: sha256_file(output_dir / n)
                    for n in (
                        "embedding_matrix.npy",
                        "embedding_manifest.json",
                        "evidence_envelope.json",
                    )
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return manifest
