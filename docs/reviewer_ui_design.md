# Reviewer UI Design

Milestone 12 implements a local Streamlit reviewer interface for the governed FastAPI research API.
The UI is a portfolio and engineering demonstrator only.

The UI architecture is deliberately thin:

- Streamlit pages collect reviewer inputs and display bounded summaries.
- A typed API client calls the existing governed FastAPI contracts.
- The UI does not import or execute segmentation, classification, or longitudinal inference code.
- Upload handling validates extension, size, NumPy safety, dimensionality, and finite values before
  API submission.
- Session state tracks last request summaries, last responses, selected evidence type, reviewer
  decisions, notes, and export status.
- Review exports are written only under the configured local output directory.

Implemented pages:

- Overview for API health, readiness, version, endpoint, enabled capabilities, limitations, and
  human-review responsibility.
- Synthetic segmentation review for bounded `.npy` volume submission, spacing metadata, predicted
  voxel count, volume, probability summary, quality findings, checksums, and output references.
- Synthetic classification review for bounded ROI submission, probabilities, threshold, synthetic
  lesion-presence engineering labels, calibration and threshold provenance, and abstention display.
- Longitudinal comparison review for previous/current masks, spacing, side, research-safe metadata,
  upstream status propagation, measurements, matching, change metrics, quality findings, and evidence
  reference.
- Evidence review for read-only governed evidence inspection by ID.
- Governance review for human engineering decisions and local export.

The UI defaults to `http://127.0.0.1:8000` and rejects remote API URLs unless explicitly enabled in
configuration. It does not start the API automatically, launch browsers in tests, make external
network calls by default, render arbitrary HTML, or log raw arrays and reviewer notes.

The interface must not be presented as clinical diagnosis, clinical decision support, radiologist
replacement, RECIST assessment, treatment-response assessment, NHS approval, medical-device
certification, or production deployment readiness.
