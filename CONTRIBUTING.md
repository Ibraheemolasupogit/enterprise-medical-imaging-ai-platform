# Contributing

This project is built milestone by milestone. Keep contributions scoped to the active milestone and avoid adding speculative implementation.

## Ground Rules

- Do not commit patient data, restricted datasets, credentials, model weights, or generated medical images.
- Use only synthetic or publicly available de-identified data in future milestones.
- Preserve the research-only intended-use boundary.
- Mark future functionality as `Planned - not yet implemented`.
- Keep implementation original; do not copy code or assets from reference repositories.

## Local Checks

```bash
python -m pip install -e ".[dev]"
make quality
```

## Pull Requests

Pull requests should describe the milestone scope, files changed, tests run, and any governance or data-safety impact.
