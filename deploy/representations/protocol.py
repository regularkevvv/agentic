"""agentic.representations.v1 request parsing and response building.

This module is the half of the handler that has no machine-learning
dependencies, so the protocol can be tested with nothing installed but a
Python interpreter. handler.py adds the model.

Nothing here logs input text. A handler runs on documents it does not own, and
an exception traceback that quotes one ends up in a deployment's log sink.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROTOCOL_VERSION = "agentic.representations.v1"
PROTOCOL_MAJOR = 1
PROTOCOL_PREFIX = "agentic.representations.v"

DENSE = "dense"
SPARSE = "sparse"
MULTI_VECTOR = "multi_vector"
KINDS = (DENSE, SPARSE, MULTI_VECTOR)

INPUT_TYPES = ("", "query", "document")

METRIC_COSINE = "cosine"
METRIC_DOT_PRODUCT = "dot_product"

# Ceilings applied before a response is built. They exist because a
# multi-vector response for a long batch is tens of millions of floats, and a
# handler that serializes one without a bound takes the endpoint down rather
# than returning an error.
DEFAULT_MAX_INPUTS = 256
DEFAULT_MAX_INPUT_BYTES = 1 << 20
DEFAULT_MAX_SPARSE_NONZERO = 65536
DEFAULT_MAX_TOKEN_VECTORS = 8192


class ProtocolError(ValueError):
    """A request or a computed response violates the protocol.

    The message names the invariant and the position. It never quotes input
    text, so it is safe to return to a caller and safe to log.
    """

    def __init__(self, invariant: str, detail: str) -> None:
        super().__init__(f"{invariant}: {detail}")
        self.invariant = invariant
        self.detail = detail


@dataclass(frozen=True)
class Request:
    """A parsed, validated request."""

    inputs: Sequence[str]
    outputs: Tuple[str, ...]
    input_type: str = ""
    truncate: Optional[bool] = None

    def wants(self, kind: str) -> bool:
        return kind in self.outputs


@dataclass(frozen=True)
class Limits:
    """Size ceilings applied to one request and its response."""

    max_inputs: int = DEFAULT_MAX_INPUTS
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES
    max_sparse_nonzero: int = DEFAULT_MAX_SPARSE_NONZERO
    max_token_vectors: int = DEFAULT_MAX_TOKEN_VECTORS


@dataclass(frozen=True)
class Space:
    """An immutable vector-space descriptor.

    ``id`` is deployment configuration, not something derived at runtime. An
    opaque endpoint cannot prove its own weights revision, so the operator who
    can is the one who declares the identity an index will be keyed on.
    """

    id: str
    provider: str
    model: str
    kind: str
    dimensions: int
    metric: str
    revision: str = ""
    tokenizer: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "model": self.model,
            "revision": self.revision,
            "tokenizer": self.tokenizer,
            "kind": self.kind,
            "dimensions": self.dimensions,
            "metric": self.metric,
        }

    def validate(self) -> None:
        if not self.id:
            raise ProtocolError("space.id", "vector space id cannot be empty")
        if not self.provider:
            raise ProtocolError("space.provider", "vector space provider cannot be empty")
        if not self.model:
            raise ProtocolError("space.model", "vector space model cannot be empty")
        if self.kind not in KINDS:
            raise ProtocolError("space.kind", f"unknown representation kind {self.kind!r}")
        if self.dimensions <= 0:
            raise ProtocolError("space.dimensions", "vector space dimensions must be positive")
        if self.metric not in (METRIC_COSINE, METRIC_DOT_PRODUCT):
            raise ProtocolError("space.metric", f"unknown similarity metric {self.metric!r}")


@dataclass
class Usage:
    """Measurements for one request."""

    input_tokens: int = 0
    request_count: int = 1
    input_bytes: int = 0
    output_bytes: int = 0

    def to_json(self) -> Dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "request_count": self.request_count,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
        }


@dataclass
class Item:
    """One input's representations."""

    dense: Optional[Sequence[float]] = None
    sparse: Optional[Dict[str, List[Any]]] = None
    multi_vector: Optional[Sequence[Sequence[float]]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(self.extra)
        if self.dense is not None:
            payload[DENSE] = list(self.dense)
        if self.sparse is not None:
            payload[SPARSE] = self.sparse
        if self.multi_vector is not None:
            payload[MULTI_VECTOR] = [list(vec) for vec in self.multi_vector]
        return payload


def check_version(version: Any) -> None:
    """Accept an additive minor bump and refuse anything else.

    An unknown major version is refused rather than parsed optimistically: a
    change large enough to bump the major is one where a wrong guess yields
    vectors that look valid and are not.
    """
    if not isinstance(version, str) or not version.startswith(PROTOCOL_PREFIX):
        raise ProtocolError("version.unknown", f"expected {PROTOCOL_VERSION}")
    digits = ""
    for char in version[len(PROTOCOL_PREFIX):]:
        if not char.isdigit():
            break
        digits += char
    if not digits:
        raise ProtocolError("version.unknown", f"expected {PROTOCOL_VERSION}")
    if int(digits) != PROTOCOL_MAJOR:
        raise ProtocolError(
            "version.major",
            f"request speaks protocol major version {int(digits)}, this handler speaks {PROTOCOL_MAJOR}",
        )


def parse_request(
    payload: Mapping[str, Any],
    supported: Iterable[str] = KINDS,
    limits: Limits = Limits(),
) -> Request:
    """Validate a request body and return it in parsed form."""
    if not isinstance(payload, Mapping):
        raise ProtocolError("request.shape", "request must be a JSON object")

    check_version(payload.get("version"))

    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ProtocolError("inputs.empty", "inputs must be a non-empty array")
    if len(inputs) > limits.max_inputs:
        raise ProtocolError(
            "inputs.limit",
            f"request has {len(inputs)} inputs, limit is {limits.max_inputs}",
        )
    for position, text in enumerate(inputs):
        if not isinstance(text, str):
            raise ProtocolError("inputs.type", f"input {position} is not a string")
        if len(text.encode("utf-8")) > limits.max_input_bytes:
            raise ProtocolError(
                "inputs.item_bytes",
                f"input {position} exceeds the {limits.max_input_bytes} byte limit",
            )

    input_type = payload.get("input_type", "")
    if input_type is None:
        input_type = ""
    if input_type not in INPUT_TYPES:
        raise ProtocolError("input_type.unknown", "input_type must be query, document, or empty")

    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ProtocolError("outputs.empty", "outputs must be a non-empty array")
    seen = set()
    supported_set = set(supported)
    for kind in outputs:
        if kind not in KINDS:
            raise ProtocolError("outputs.unknown", f"unknown representation kind {kind!r}")
        if kind in seen:
            raise ProtocolError("outputs.duplicate", f"{kind} is listed more than once")
        if kind not in supported_set:
            raise ProtocolError("outputs.unsupported", f"this endpoint does not produce {kind}")
        seen.add(kind)

    truncate = payload.get("truncate")
    if truncate is not None and not isinstance(truncate, bool):
        raise ProtocolError("truncate.type", "truncate must be a boolean")

    return Request(
        inputs=inputs,
        outputs=tuple(outputs),
        input_type=input_type,
        truncate=truncate,
    )


def canonical_sparse(
    weights: Mapping[Any, Any],
    vocabulary: int,
    limits: Limits = Limits(),
) -> Dict[str, List[Any]]:
    """Turn a token-weight mapping into canonical coordinate form.

    Indices come out strictly increasing, which makes the value canonical: one
    input has exactly one encoding, so fixtures compare and duplicates are
    detectable rather than resolved by whichever key was written last. Exact
    zeros are dropped, because a zero coordinate is one that should have been
    omitted.
    """
    coordinates: Dict[int, float] = {}
    for raw_index, raw_value in weights.items():
        index = int(raw_index)
        value = float(raw_value)
        if index < 0 or index >= vocabulary:
            raise ProtocolError(
                "sparse.range",
                f"coordinate {index} is outside the declared vocabulary of {vocabulary}",
            )
        if not math.isfinite(value):
            raise ProtocolError("sparse.finite", f"weight at coordinate {index} is not finite")
        if value == 0.0:
            continue
        if index in coordinates:
            raise ProtocolError("sparse.duplicate", f"coordinate {index} appears more than once")
        coordinates[index] = value

    if len(coordinates) > limits.max_sparse_nonzero:
        raise ProtocolError(
            "sparse.limit",
            f"{len(coordinates)} nonzero coordinates exceed the limit of {limits.max_sparse_nonzero}",
        )

    indices = sorted(coordinates)
    return {"indices": indices, "values": [coordinates[i] for i in indices]}


def check_multi_vector(vectors: Sequence[Sequence[float]], dimensions: int, position: int,
                       limits: Limits = Limits()) -> None:
    """Reject a token multi-vector that no consumer could store."""
    if not vectors:
        raise ProtocolError("multi_vector.empty", f"item {position} has no token vectors")
    if len(vectors) > limits.max_token_vectors:
        raise ProtocolError(
            "multi_vector.limit",
            f"item {position} has {len(vectors)} token vectors, limit is {limits.max_token_vectors}",
        )
    for token, vector in enumerate(vectors):
        if len(vector) != dimensions:
            raise ProtocolError(
                "multi_vector.width",
                f"item {position} token {token} has width {len(vector)}, space declares {dimensions}",
            )


def build_response(
    model: str,
    spaces: Mapping[str, Space],
    items: Sequence[Item],
    request: Request,
    usage: Optional[Usage] = None,
) -> Dict[str, Any]:
    """Assemble a response and check it against the request before returning.

    A partially correct batch is worse than an error: written into an index,
    the damage is silent. Everything here fails the whole response instead.
    """
    if len(items) != len(request.inputs):
        raise ProtocolError(
            "data.cardinality",
            f"built {len(items)} items for {len(request.inputs)} inputs",
        )

    payload_spaces: Dict[str, Any] = {}
    for kind in request.outputs:
        space = spaces.get(kind)
        if space is None:
            raise ProtocolError("spaces.missing", f"no vector space configured for {kind}")
        space.validate()
        if space.kind != kind:
            raise ProtocolError("spaces.kind", f"space for {kind} declares kind {space.kind!r}")
        payload_spaces[kind] = space.to_json()

    data = []
    for position, item in enumerate(items):
        encoded = item.to_json()
        for kind in KINDS:
            present = kind in encoded
            if present and not request.wants(kind):
                raise ProtocolError(
                    "data.unrequested",
                    f"item {position} carries {kind}, which was not requested",
                )
            if request.wants(kind) and not present:
                raise ProtocolError(
                    "data.missing",
                    f"item {position} is missing the requested {kind}",
                )
        _check_item(encoded, request, spaces, position)
        data.append(encoded)

    if usage is None:
        usage = Usage()
    if usage.input_bytes == 0:
        usage.input_bytes = sum(len(text.encode("utf-8")) for text in request.inputs)

    return {
        "version": PROTOCOL_VERSION,
        "model": model,
        "spaces": payload_spaces,
        "data": data,
        "usage": usage.to_json(),
    }


def _check_item(encoded: Mapping[str, Any], request: Request,
                spaces: Mapping[str, Space], position: int) -> None:
    if request.wants(DENSE):
        vector = encoded[DENSE]
        dimensions = spaces[DENSE].dimensions
        if len(vector) != dimensions:
            raise ProtocolError(
                "dense.width",
                f"item {position} has width {len(vector)}, space declares {dimensions}",
            )
        for offset, value in enumerate(vector):
            if not math.isfinite(value):
                raise ProtocolError("dense.finite", f"item {position} value {offset} is not finite")

    if request.wants(SPARSE):
        sparse = encoded[SPARSE]
        indices = sparse["indices"]
        values = sparse["values"]
        if len(indices) != len(values):
            raise ProtocolError(
                "sparse.pairs",
                f"item {position} has {len(indices)} indices and {len(values)} values",
            )
        previous = -1
        for offset, index in enumerate(indices):
            if index <= previous:
                raise ProtocolError(
                    "sparse.order",
                    f"item {position} indices are not strictly increasing at {offset}",
                )
            previous = index

    if request.wants(MULTI_VECTOR):
        check_multi_vector(encoded[MULTI_VECTOR], spaces[MULTI_VECTOR].dimensions, position)


def dumps(payload: Mapping[str, Any]) -> str:
    """Serialize a response with stable key order, for golden comparisons."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
