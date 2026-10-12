"""Reviewer page for governed M22 embedding retrieval evidence."""

from __future__ import annotations

from typing import Any


def render(st_module: Any, response: dict[str, Any] | None = None) -> None:
    """Render retrieval evidence using the existing Streamlit surface."""
    st_module.title("Embedding retrieval")
    st_module.warning("Embedding-space similarity is not diagnostic similarity.")
    st_module.caption("Research and engineering use only; human interpretation is required.")
    if response is None:
        st_module.info("Submit a governed embedding search to review ranked results.")
        return
    st_module.write(
        {
            "query": response.get("query"),
            "encoder_family": response.get("encoder_family"),
            "checkpoint_checksum": response.get("encoder_checkpoint_checksum"),
            "index_checksum": response.get("index_checksum"),
            "latency_ms": response.get("latency_ms"),
        }
    )
    st_module.subheader("Ranked results")
    st_module.dataframe(response.get("results", []), use_container_width=True)
    for warning in response.get("warnings", []):
        st_module.caption(str(warning))
