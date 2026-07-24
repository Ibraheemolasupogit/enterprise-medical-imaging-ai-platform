"""Synthetic localisation fixtures with engineering adrenal-region placeholders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medical_imaging_platform.registration.fixtures import _write_preprocessed_fixture


def generate_localisation_fixture(
    output_dir: Path,
    *,
    volume_shape: tuple[int, int, int] = (32, 32, 32),
    left_centre: tuple[int, int, int] = (16, 12, 10),
    right_centre: tuple[int, int, int] = (16, 12, 22),
    roi_radius_voxels: int = 3,
    translation: tuple[int, int, int] = (0, 0, 0),
    noise_std: float = 0.0,
    random_seed: int = 20260724,
    overwrite: bool = False,
) -> Path:
    """Generate one preprocessing-compatible localisation fixture directory."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Localisation fixture directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)
    left = _mask(volume_shape, _translate(left_centre, translation), roi_radius_voxels)
    right = _mask(volume_shape, _translate(right_centre, translation), roi_radius_voxels)
    if not np.any(left) or not np.any(right):
        raise ValueError("Synthetic localisation masks must remain non-empty.")
    if np.any(left & right):
        raise ValueError("Synthetic left and right adrenal-region masks must not overlap.")
    volume = rng.normal(0.0, noise_std, size=volume_shape).astype(np.float32)
    volume[left] = 1.0
    volume[right] = 0.85
    _write_preprocessed_fixture(output_dir, volume, run_id="preprocess-localisation-synthetic")
    with (output_dir / "left_adrenal_mask.npy").open("wb") as handle:
        np.save(handle, left.astype(np.uint8))
    with (output_dir / "right_adrenal_mask.npy").open("wb") as handle:
        np.save(handle, right.astype(np.uint8))
    metadata_path = output_dir / "localisation_fixture_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "fixture_type": "synthetic engineering adrenal-region placeholder",
                "volume_shape": list(volume_shape),
                "left_centre_voxel": list(_translate(left_centre, translation)),
                "right_centre_voxel": list(_translate(right_centre, translation)),
                "roi_radius_voxels": roi_radius_voxels,
                "translation_zyx": list(translation),
                "random_seed": random_seed,
                "labels_are_clinical": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir


def _translate(
    centre: tuple[int, int, int], translation: tuple[int, int, int]
) -> tuple[int, int, int]:
    return (
        int(centre[0] + translation[0]),
        int(centre[1] + translation[1]),
        int(centre[2] + translation[2]),
    )


def _mask(shape: tuple[int, int, int], centre: tuple[int, int, int], radius: int) -> np.ndarray:
    z, y, x = np.indices(shape)
    mask = ((z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2) <= radius**2
    return np.asarray(mask, dtype=bool)
