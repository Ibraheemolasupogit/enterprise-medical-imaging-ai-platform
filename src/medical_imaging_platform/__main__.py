"""Module entry point."""

from __future__ import annotations

import sys

_M22_COMMANDS = {
    "generate-embeddings",
    "query-similar-studies",
    "evaluate-retrieval",
    "run-embedding-benchmark",
    "validate-embedding-index",
}

if sys.argv[1:2] and sys.argv[1] in _M22_COMMANDS:
    from medical_imaging_platform.embeddings.cli import main
else:
    from medical_imaging_platform.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
