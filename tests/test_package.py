import medical_imaging_platform
from medical_imaging_platform import __version__


def test_package_import() -> None:
    assert medical_imaging_platform is not None


def test_package_version() -> None:
    assert __version__ == "0.1.0"
