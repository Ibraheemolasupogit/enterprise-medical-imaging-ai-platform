# Documentation index

| Area | Documents |
|---|---|
| Architecture | [Current architecture](architecture.md), [diagram/status](architecture/final_end_to_end_architecture.md), [data flow](data_flow.md) — actual dependencies and boundaries |
| Data/DICOM/preprocessing | [Synthetic data](synthetic_data_design.md), [provenance](data_provenance.md), [DICOM ingestion](dicom_ingestion_design.md), [de-identification](deidentification_design.md), [QC](imaging_quality_control.md), [preprocessing](preprocessing_design.md), [M19 baseline](preprocessing_baseline.md) |
| ML task families | [Localisation](localisation_design.md), [segmentation](segmentation_design.md), [classification/CNN–ViT](classification_design.md), [registration](registration_design.md), [longitudinal](longitudinal_analysis_design.md) |
| Detection | [Axial detectors](detection_design.md) — reference boxes, model families and metrics |
| Embeddings/retrieval | [Index/search contracts](embeddings.md) — policy, stored validation and limitations |
| API | [Design/configuration](api_design.md), [route contracts](api_contracts.md) |
| Reviewer UI | [Design](reviewer_ui_design.md), [workflow](reviewer_workflow.md) — overlays, retrieval and separate decisions |
| Governance | [Intended use](../governance/intended_use.md), [excluded use](../governance/excluded_use.md), [human oversight](../governance/human_oversight.md), [change control](../governance/model_change_control.md), [hazards](../governance/hazard_log.csv) |
| Evidence | [Common envelopes](evidence_contracts.md), [model comparisons](engineering_comparison.md), [release evidence](release_evidence.md), [technical summary](portfolio/project_summary.md) |
| Operations/deployment | [Security](security.md), [containers](containerisation.md), [Kubernetes](deployment/kubernetes.md), [AWS target state](deployment/aws_architecture.md), [observability](operations/observability.md), [runbooks](operations/runbooks.md), [recovery](operations/rollback_recovery.md) |
| Demo | [Canonical CPU demonstration](platform_demo.md), [local API/UI startup](local_startup.md), [workflow commands](portfolio/demonstration_script.md) |
| Limitations | [Platform limitations](limitations.md), [governance limitations](../governance/limitations.md), [current roadmap](roadmap.md) |

Implemented, executed/validated, simulated, static design and target state are distinct labels.
Generated evidence remains ignored; reproduction commands provide the authoritative run context.
