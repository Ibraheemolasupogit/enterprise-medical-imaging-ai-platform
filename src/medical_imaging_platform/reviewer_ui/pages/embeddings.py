"""Reviewer page for governed M22 embedding retrieval evidence."""

from __future__ import annotations

from typing import Any

from medical_imaging_platform.reviewer_ui.api_client import ReviewerAPIClient
from medical_imaging_platform.reviewer_ui.components.provenance import render_provenance
from medical_imaging_platform.reviewer_ui.models import ReviewerAPIError, ReviewerUIConfig
from medical_imaging_platform.reviewer_ui.state import remember_response


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


def render_search(st_module: Any, config: ReviewerUIConfig, client: ReviewerAPIClient) -> None:
    """Use explicitly configured index choices or a governed server-side path."""
    try:
        entries = client.embedding_indexes().get("indexes", [])
        available = [entry for entry in entries if entry["status"] == "available"]
        if available:
            directory = st_module.selectbox(
                "Embedding index", [entry["index_directory"] for entry in available]
            )
            selected = next(entry for entry in available if entry["index_directory"] == directory)
            samples = selected["sample_ids"]
            sample_id = (
                st_module.selectbox("Query sample", samples)
                if samples
                else st_module.text_input("Query sample ID")
            )
        else:
            st_module.info(
                "No available index configured. Supply a path under the API's allowed input roots."
            )
            directory = st_module.text_input("Embedding index directory")
            sample_id = st_module.text_input("Query sample ID")
        top_k = int(
            st_module.number_input(
                "Top K",
                min_value=1,
                max_value=min(100, config.maximum_review_items),
                value=min(5, config.maximum_review_items),
                step=1,
            )
        )
        if st_module.button("Search similar studies") and directory and sample_id:
            response = client.search_embeddings(
                {"index_directory": directory, "sample_id": sample_id, "top_k": top_k}
            )
            remember_response(
                st_module.session_state, {"page": "embeddings", "sample_id": sample_id}, response
            )
            st_module.session_state["embedding_response"] = response
        response = st_module.session_state.get("embedding_response")
        render(st_module, response)
        if isinstance(response, dict):
            render_provenance(
                st_module,
                {
                    "artifact": response.get("artifact"),
                    "query": response.get("query"),
                    "retrieval_policy": response.get("retrieval_policy"),
                },
            )
    except ReviewerAPIError as exc:
        st_module.error(str(exc))
        render(st_module)
