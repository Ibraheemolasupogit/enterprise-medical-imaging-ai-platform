"""Research-fixture de-identification policy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Action = Literal["REMOVE", "REPLACE", "HASH", "KEEP_WITH_JUSTIFICATION"]

DIRECT_IDENTIFIER_KEYWORDS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "AccessionNumber",
    "StudyID",
    "OtherPatientIDs",
    "OtherPatientNames",
)


class DeidentificationPolicy(BaseModel):
    """Direct identifier actions and UID remapping settings."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(min_length=1)
    uid_root: str = Field(min_length=1)
    patient_id_prefix: str = Field(min_length=1)
    direct_identifier_actions: dict[str, Action]


def default_policy(
    *,
    policy_version: str,
    uid_root: str,
    patient_id_prefix: str,
) -> DeidentificationPolicy:
    """Create the default Milestone 3 research de-identification policy."""
    actions: dict[str, Action] = dict.fromkeys(DIRECT_IDENTIFIER_KEYWORDS, "REMOVE")
    actions["PatientID"] = "REPLACE"
    actions["PatientName"] = "REPLACE"
    return DeidentificationPolicy(
        policy_version=policy_version,
        uid_root=uid_root.rstrip("."),
        patient_id_prefix=patient_id_prefix,
        direct_identifier_actions=actions,
    )
