import json
import logging
import sys

from medical_imaging_platform.utils.logging import JsonLogFormatter, configure_logging


def test_structured_logging_initialisation() -> None:
    logger = configure_logging("DEBUG")

    assert logger.level == logging.DEBUG
    assert logger.handlers
    assert isinstance(logger.handlers[0].formatter, JsonLogFormatter)


def test_json_serialisable_log_output() -> None:
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["message"] == "hello"
    assert "timestamp" in payload


def test_json_log_output_with_exception() -> None:
    formatter = JsonLogFormatter()
    try:
        raise RuntimeError("example")
    except RuntimeError:
        record = logging.getLogger("test").makeRecord(
            name="test",
            level=logging.ERROR,
            fn=__file__,
            lno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
            func=None,
            extra=None,
        )

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "ERROR"
    assert "RuntimeError: example" in payload["exception"]
