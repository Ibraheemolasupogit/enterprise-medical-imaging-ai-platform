import json
from pathlib import Path

import numpy as np
import pytest

from medical_imaging_platform.embeddings.core import (
    EmbeddingIndex,
    l2_normalize,
    sha256_file,
    validate_index,
)


def r(s, e):
    return {"sample_id": s, "source_sample_id": s, "embedding_id": e, "scenario": "s1", "label": 1}


def w(d, rs):
    d = Path(d)
    (d / "evidence").mkdir(parents=True)
    p = d / "embedding_matrix.npy"
    np.save(p, np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    (d / "embedding_manifest.json").write_text(
        json.dumps(
            {
                "embedding_dimension": 2,
                "records": rs,
                "encoder_family": "small_3d_cnn",
                "matrix_checksum": sha256_file(p),
                "normalization": "l2",
                "encoder_checkpoint_checksum": sha256_file(p),
                "source_dataset_manifest_checksum": sha256_file(p),
                "config_checksum": sha256_file(p),
            }
        )
    )
    (d / "evidence" / "checksum_manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    name: sha256_file(d / name)
                    for name in ("embedding_matrix.npy", "embedding_manifest.json")
                }
            }
        )
    )


def test_norm():
    x = l2_normalize(np.asarray([[3, 4]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(x), 1)
    with pytest.raises(ValueError):
        l2_normalize(np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        l2_normalize(np.asarray([[np.nan, 1]], dtype=np.float32))


def test_index(tmp_path):
    d = tmp_path / "i"
    w(d, [r("a", "ea"), r("b", "eb")])
    i = EmbeddingIndex.load(d)
    assert i.dimension == 2
    assert i.search([1, 0], 2)
    assert validate_index(d)["valid"]
    np.save(d / "embedding_matrix.npy", np.asarray([[2, 0], [0, 1]], dtype=np.float32))
    with pytest.raises(ValueError):
        EmbeddingIndex.load(d)


def test_cli_parser():
    from medical_imaging_platform.embeddings.cli import _parser

    assert (
        _parser().parse_args(["validate-embedding-index", "x"]).command
        == "validate-embedding-index"
    )
