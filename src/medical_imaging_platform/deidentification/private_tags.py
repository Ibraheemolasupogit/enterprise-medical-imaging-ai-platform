"""Private-tag detection and removal."""

from __future__ import annotations

from pydicom.dataset import FileDataset


def count_private_tags(dataset: FileDataset) -> int:
    """Count private DICOM tags."""
    return sum(1 for element in dataset.iterall() if element.tag.is_private)


def remove_private_tags(dataset: FileDataset) -> int:
    """Remove all private tags and return the number removed."""
    count = count_private_tags(dataset)
    dataset.remove_private_tags()
    return count
