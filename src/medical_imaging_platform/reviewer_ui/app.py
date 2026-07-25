"""Streamlit reviewer UI entry point."""

from __future__ import annotations

from pathlib import Path

from medical_imaging_platform.reviewer_ui.api_client import ReviewerAPIClient
from medical_imaging_platform.reviewer_ui.components.disclaimers import render_disclaimer
from medical_imaging_platform.reviewer_ui.config import load_reviewer_ui_config
from medical_imaging_platform.reviewer_ui.pages import (
    classification,
    evidence,
    governance,
    longitudinal,
    overview,
    segmentation,
)
from medical_imaging_platform.reviewer_ui.security import enforce_loopback_url
from medical_imaging_platform.reviewer_ui.state import initialise_state


def run_reviewer_app(config_path: Path | None = None) -> None:
    """Run the local reviewer-facing Streamlit app."""
    import streamlit as st

    config = load_reviewer_ui_config(config_path or Path("config/reviewer_ui.yaml"))
    enforce_loopback_url(config.api_base_url, allow_remote=config.allow_remote_api)
    st.set_page_config(
        page_title=config.page_title,
        page_icon=config.page_icon,
        layout=config.layout,
    )
    initialise_state(st.session_state)
    render_disclaimer(st)
    client = ReviewerAPIClient(config)
    pages = ["Overview"]
    if config.enable_segmentation_page:
        pages.append("Segmentation")
    if config.enable_classification_page:
        pages.append("Classification")
    if config.enable_longitudinal_page:
        pages.append("Longitudinal")
    if config.enable_evidence_page:
        pages.append("Evidence")
    if config.enable_governance_page:
        pages.append("Governance")
    selected = st.sidebar.radio("Reviewer pages", pages)
    if selected == "Overview":
        overview.render(st, config, client)
    elif selected == "Segmentation":
        segmentation.render(st, config, client)
    elif selected == "Classification":
        classification.render(st, config, client)
    elif selected == "Longitudinal":
        longitudinal.render(st, config, client)
    elif selected == "Evidence":
        evidence.render(st, config, client)
    else:
        governance.render(st, config)


if __name__ == "__main__":
    run_reviewer_app()
