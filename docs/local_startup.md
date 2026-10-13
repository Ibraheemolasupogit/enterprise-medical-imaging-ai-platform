# Local API and reviewer startup

Install `requirements/ci-quality.txt`, then run `make platform-demo` from the repository root. It
writes checkpoints, validated indexes and explicit runtime configurations under
`reports/generated/platform-demo/`. These paths remain ignored by Git.

## Start

Use two terminals, both in the repository root:

```bash
python3 -m medical_imaging_platform serve-api --config reports/generated/platform-demo/config/api.yaml
python3 -m medical_imaging_platform serve-reviewer-ui --config reports/generated/platform-demo/config/reviewer_ui.yaml
```

API uses loopback port 8000; reviewer UI uses loopback port 8501. The UI calls the governed API and
never imports model inference. Choose Embeddings / Similarity Retrieval, a configured encoder index,
a query sample and K. Ranked results include scores, query provenance and checkpoint/index hashes.

For segmentation, upload a prepared `image.npy` from `segmentation-data/`, retain its recorded
spacing, choose an axial slice and submit. The source slice and predicted-mask overlay use fixed
opacity 0.35. Probability summaries, warnings, source/model hashes and evidence reference are visible.
Changing slice requires a new submission. No editing or prediction overwrite occurs.

## Health and readiness

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ready
curl -fsS http://127.0.0.1:8000/v1/embeddings/indexes
```

`/health` reports process liveness. `/ready` verifies configured artifact existence/index integrity,
reports capability availability and distinguishes degraded service with no configured model artifacts.
Unconfigured optional capabilities remain unavailable; a healthy process is not proof they can infer.
Configured missing/tampered indexes fail readiness. Model configuration/checkpoint mismatch is also
rejected at inference. Default `config/api.yaml` intentionally supplies no checkpoints or indexes.

Input/index paths are governed by `allowed_input_roots`; evidence paths by `allowed_evidence_roots`.
The generated configuration restricts them to its demo root. Index listing enumerates explicit paths,
never searches arbitrary directories. Array sizes, preview shape and slice bounds are checked.

## Shutdown and cleanup

Press Ctrl-C in both service terminals. Do not delete artifacts while services or reviews use them.
Choose a new output directory for repeat runs. To clean up, remove only the chosen generated demo
folder after preserving required evidence; never remove source/configuration directories. No cloud
resources are created and no external model downloads are required.
