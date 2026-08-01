"""SageMaker inference entrypoint wrapping the shared handler.

SageMaker's Python inference toolkit calls four functions rather than one
callable. They are thin: the protocol, the model, and every check live in
handler.py, so a Hugging Face endpoint and a SageMaker endpoint behave
identically and are tested once.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from handler import EndpointHandler
from protocol import ProtocolError

JSON_CONTENT_TYPE = "application/json"


def model_fn(model_dir: str, context: Any = None) -> EndpointHandler:
    """Load the model once, when the container starts."""
    return EndpointHandler(model_dir)


def input_fn(request_body: Any, request_content_type: str = JSON_CONTENT_TYPE) -> Dict[str, Any]:
    """Parse the request body. Only JSON is accepted."""
    if request_content_type and not request_content_type.startswith(JSON_CONTENT_TYPE):
        raise ProtocolError(
            "content_type.unsupported",
            f"expected {JSON_CONTENT_TYPE}, got {request_content_type!r}",
        )
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    if isinstance(request_body, str):
        return json.loads(request_body)
    return request_body


def predict_fn(data: Dict[str, Any], handler: EndpointHandler) -> Dict[str, Any]:
    return handler(data)


def output_fn(prediction: Dict[str, Any], accept: str = JSON_CONTENT_TYPE) -> Tuple[str, str]:
    if accept and not accept.startswith(JSON_CONTENT_TYPE) and accept != "*/*":
        raise ProtocolError(
            "accept.unsupported",
            f"expected {JSON_CONTENT_TYPE}, got {accept!r}",
        )
    return json.dumps(prediction), JSON_CONTENT_TYPE
