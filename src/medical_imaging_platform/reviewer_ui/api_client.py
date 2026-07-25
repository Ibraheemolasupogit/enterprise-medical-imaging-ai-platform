"""Typed client for the governed FastAPI service."""

from __future__ import annotations

import uuid
from typing import Any

import httpx

from medical_imaging_platform.reviewer_ui.models import ReviewerAPIError, ReviewerUIConfig


class ReviewerAPIClient:
    """Small no-retry API client used by the reviewer UI."""

    def __init__(
        self,
        config: ReviewerUIConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._client = client or httpx.Client(  # nosec B113
            base_url=config.api_base_url,
            timeout=float(config.request_timeout_seconds),
        )

    def health(self, request_id: str | None = None) -> dict[str, Any]:
        return self._request("GET", "/health", request_id=request_id)

    def ready(self, request_id: str | None = None) -> dict[str, Any]:
        return self._request("GET", "/ready", request_id=request_id)

    def version(self, request_id: str | None = None) -> dict[str, Any]:
        return self._request("GET", "/version", request_id=request_id)

    def predict_segmentation(
        self, payload: dict[str, Any], request_id: str | None = None
    ) -> dict[str, Any]:
        return self._request(
            "POST", "/v1/segmentation/predict", json=payload, request_id=request_id
        )

    def predict_classification(
        self, payload: dict[str, Any], request_id: str | None = None
    ) -> dict[str, Any]:
        return self._request(
            "POST", "/v1/classification/predict", json=payload, request_id=request_id
        )

    def analyse_longitudinal(
        self, payload: dict[str, Any], request_id: str | None = None
    ) -> dict[str, Any]:
        return self._request(
            "POST", "/v1/longitudinal/analyse", json=payload, request_id=request_id
        )

    def review_segmentation(
        self, experiment_id: str, request_id: str | None = None
    ) -> dict[str, Any]:
        return self._request(
            "GET", f"/v1/review/segmentation/{experiment_id}", request_id=request_id
        )

    def review_classification(
        self, experiment_id: str, request_id: str | None = None
    ) -> dict[str, Any]:
        return self._request(
            "GET", f"/v1/review/classification/{experiment_id}", request_id=request_id
        )

    def review_longitudinal(
        self, analysis_id: str, request_id: str | None = None
    ) -> dict[str, Any]:
        return self._request("GET", f"/v1/review/longitudinal/{analysis_id}", request_id=request_id)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        rid = request_id or f"review-ui-{uuid.uuid4().hex[:16]}"
        try:
            response = self._client.request(method, path, json=json, headers={"X-Request-ID": rid})
        except httpx.HTTPError as exc:
            raise ReviewerAPIError(
                503,
                "UI-API-UNAVAILABLE",
                "The governed API is unavailable.",
                request_id=rid,
            ) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise ReviewerAPIError(
                response.status_code,
                "UI-API-BAD-RESPONSE",
                "The governed API returned an invalid response.",
                request_id=rid,
            ) from exc
        if response.status_code >= 400:
            raise ReviewerAPIError(
                response.status_code,
                str(payload.get("error_code", "UI-API-ERROR")),
                _sanitise_message(str(payload.get("message", "The API request failed."))),
                request_id=str(payload.get("request_id", rid)),
                details=_string_details(payload.get("details", {})),
            )
        if isinstance(payload, dict):
            payload.setdefault("request_id", response.headers.get("X-Request-ID", rid))
            return payload
        raise ReviewerAPIError(
            response.status_code,
            "UI-API-BAD-RESPONSE",
            "The governed API returned a non-object response.",
            request_id=rid,
        )


def _sanitise_message(message: str) -> str:
    if "/" in message or "\\" in message:
        return "The API request failed. Review configured inputs and evidence identifiers."
    return message


def _string_details(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}
