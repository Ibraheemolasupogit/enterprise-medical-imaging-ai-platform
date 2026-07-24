"""Evidence export and validation for longitudinal analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from medical_imaging_platform.longitudinal.report import render_longitudinal_report
from medical_imaging_platform.synthetic.manifest import sha256_file
from medical_imaging_platform.utils.exceptions import MedicalImagingPlatformError


class LongitudinalOutputError(MedicalImagingPlatformError):
    """Raised when longitudinal evidence cannot be validated."""


def write_longitudinal_evidence(output_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    """Write stable JSON, Markdown, and checksum evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "pair_manifest": output_dir / "pair_manifest.json",
        "lesion_measurements": output_dir / "lesion_measurements.json",
        "lesion_matches": output_dir / "lesion_matches.json",
        "longitudinal_changes": output_dir / "longitudinal_changes.json",
        "quality_findings": output_dir / "quality_findings.json",
        "longitudinal_summary": output_dir / "longitudinal_summary.json",
        "longitudinal_report": output_dir / "longitudinal_report.md",
    }
    _write_json(files["pair_manifest"], payload["pair_manifest"])
    _write_json(files["lesion_measurements"], payload["lesion_measurements"])
    _write_json(files["lesion_matches"], payload["lesion_matches"])
    _write_json(files["longitudinal_changes"], payload["longitudinal_changes"])
    _write_json(files["quality_findings"], payload["quality_findings"])
    _write_json(files["longitudinal_summary"], payload["summary"])
    _write_text(files["longitudinal_report"], render_longitudinal_report(payload))
    return {key: sha256_file(path) for key, path in files.items()}


def write_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(output_dir / "analysis_manifest.json", payload)


def validate_longitudinal_analysis(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "analysis_manifest.json"
    if not manifest_path.exists():
        raise LongitudinalOutputError("Longitudinal analysis manifest is missing.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LongitudinalOutputError(f"Longitudinal manifest is invalid JSON: {exc}") from exc
    for key, relative_path in manifest["paths"].items():
        path = output_dir / relative_path
        if not path.exists():
            raise LongitudinalOutputError(f"Missing longitudinal artefact: {key}")
        expected = manifest["checksums"].get(key)
        if expected is not None and sha256_file(path) != expected:
            raise LongitudinalOutputError(f"Checksum mismatch for {key}")
    return cast(dict[str, Any], manifest)


def inspect_longitudinal_analysis(output_dir: Path) -> dict[str, Any]:
    manifest = validate_longitudinal_analysis(output_dir)
    return {
        "analysis_id": manifest["analysis_id"],
        "status": manifest["status"],
        "case_id": manifest["pair_manifest"]["case_id"],
        "side": manifest["pair_manifest"]["side"],
        "engineering_labels": manifest["summary"]["engineering_labels"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
