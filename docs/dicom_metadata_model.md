# DICOM Metadata Model

The Milestone 3 metadata model exposes technical imaging metadata only.

Included fields:

- Study, series, and SOP UIDs.
- Modality.
- Body part examined.
- Study date.
- Series description.
- Manufacturer and model.
- Rows and columns.
- Pixel spacing and slice thickness.
- Orientation and position.
- Instance number and slice location.
- Rescale slope and intercept.
- Photometric interpretation.
- Bit depth and pixel representation.
- Transfer syntax UID.
- Burned-in annotation status.
- Pixel-data presence.
- Private-tag count.

Direct identifier values such as patient name, patient ID, accession number, institution, address, and physician/operator names are not exposed in normal metadata output.
