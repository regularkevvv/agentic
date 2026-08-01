"""Contract tests for the agentic.representations.v1 handler.

These need only pytest and jsonschema; the model is faked, so the protocol can
be verified on a laptop and in CI without a GPU or a multi-gigabyte download.

Run:  pip install -r requirements-dev.txt && pytest
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping

import pytest

from handler import Config, EndpointHandler
from protocol import (
    DENSE,
    MULTI_VECTOR,
    PROTOCOL_VERSION,
    SPARSE,
    Item,
    Limits,
    ProtocolError,
    Request,
    Space,
    Usage,
    build_response,
    canonical_sparse,
    check_version,
    parse_request,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_DIR = os.path.join(HERE, "schema")
TESTDATA_DIR = os.path.join(HERE, "testdata")


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def request_schema() -> Dict[str, Any]:
    return load_json(os.path.join(SCHEMA_DIR, "request.schema.json"))


@pytest.fixture(scope="module")
def response_schema() -> Dict[str, Any]:
    return load_json(os.path.join(SCHEMA_DIR, "response.schema.json"))


@pytest.fixture(scope="module")
def golden_request() -> Dict[str, Any]:
    return load_json(os.path.join(TESTDATA_DIR, "request.json"))


@pytest.fixture(scope="module")
def golden_response() -> Dict[str, Any]:
    return load_json(os.path.join(TESTDATA_DIR, "response.json"))


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def test_golden_request_matches_schema(request_schema, golden_request):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(golden_request, request_schema)


def test_golden_response_matches_schema(response_schema, golden_response):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(golden_response, response_schema)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda body: body.pop("inputs"), id="no inputs"),
        pytest.param(lambda body: body.update(outputs=[]), id="no outputs"),
        pytest.param(lambda body: body.update(outputs=["colbert"]), id="unknown kind"),
        pytest.param(lambda body: body.update(outputs=["dense", "dense"]), id="duplicate kind"),
        pytest.param(lambda body: body.update(input_type="passage"), id="unknown input type"),
        pytest.param(lambda body: body.update(version="agentic.representations.v2"), id="wrong major"),
    ],
)
def test_schema_rejects_malformed_requests(request_schema, golden_request, mutate):
    jsonschema = pytest.importorskip("jsonschema")
    body = json.loads(json.dumps(golden_request))
    mutate(body)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(body, request_schema)


def test_schema_ignores_additive_fields(request_schema, response_schema, golden_request, golden_response):
    jsonschema = pytest.importorskip("jsonschema")
    request = json.loads(json.dumps(golden_request))
    request["future_field"] = {"anything": True}
    jsonschema.validate(request, request_schema)

    response = json.loads(json.dumps(golden_response))
    response["future_field"] = [1, 2, 3]
    jsonschema.validate(response, response_schema)


# --------------------------------------------------------------------------
# Version
# --------------------------------------------------------------------------


def test_check_version_accepts_additive_minor():
    check_version(PROTOCOL_VERSION)
    check_version("agentic.representations.v1.4")


@pytest.mark.parametrize("version", ["", None, "v1", "agentic.representations.vX", 1])
def test_check_version_rejects_unknown(version):
    with pytest.raises(ProtocolError):
        check_version(version)


def test_check_version_rejects_other_major():
    with pytest.raises(ProtocolError) as excinfo:
        check_version("agentic.representations.v2")
    assert excinfo.value.invariant == "version.major"


# --------------------------------------------------------------------------
# Request parsing
# --------------------------------------------------------------------------


def test_parse_request_returns_parsed_form(golden_request):
    request = parse_request(golden_request)
    assert list(request.inputs) == ["a document", "another document"]
    assert request.outputs == ("dense", "sparse")
    assert request.input_type == "document"
    assert request.truncate is True
    assert request.wants(DENSE) and not request.wants(MULTI_VECTOR)


@pytest.mark.parametrize(
    "body,invariant",
    [
        ({"version": PROTOCOL_VERSION, "outputs": ["dense"]}, "inputs.empty"),
        ({"version": PROTOCOL_VERSION, "inputs": [], "outputs": ["dense"]}, "inputs.empty"),
        ({"version": PROTOCOL_VERSION, "inputs": [1], "outputs": ["dense"]}, "inputs.type"),
        ({"version": PROTOCOL_VERSION, "inputs": ["a"]}, "outputs.empty"),
        ({"version": PROTOCOL_VERSION, "inputs": ["a"], "outputs": ["x"]}, "outputs.unknown"),
        ({"version": PROTOCOL_VERSION, "inputs": ["a"], "outputs": ["dense", "dense"]}, "outputs.duplicate"),
        (
            {"version": PROTOCOL_VERSION, "inputs": ["a"], "outputs": ["dense"], "input_type": "passage"},
            "input_type.unknown",
        ),
        (
            {"version": PROTOCOL_VERSION, "inputs": ["a"], "outputs": ["dense"], "truncate": "yes"},
            "truncate.type",
        ),
    ],
)
def test_parse_request_rejects_malformed(body, invariant):
    with pytest.raises(ProtocolError) as excinfo:
        parse_request(body)
    assert excinfo.value.invariant == invariant


def test_parse_request_rejects_unsupported_output():
    body = {"version": PROTOCOL_VERSION, "inputs": ["a"], "outputs": ["sparse"]}
    with pytest.raises(ProtocolError) as excinfo:
        parse_request(body, supported=(DENSE,))
    assert excinfo.value.invariant == "outputs.unsupported"


def test_parse_request_enforces_limits():
    limits = Limits(max_inputs=1, max_input_bytes=4)
    with pytest.raises(ProtocolError) as excinfo:
        parse_request(
            {"version": PROTOCOL_VERSION, "inputs": ["a", "b"], "outputs": ["dense"]},
            limits=limits,
        )
    assert excinfo.value.invariant == "inputs.limit"

    with pytest.raises(ProtocolError) as excinfo:
        parse_request(
            {"version": PROTOCOL_VERSION, "inputs": ["far too long"], "outputs": ["dense"]},
            limits=limits,
        )
    assert excinfo.value.invariant == "inputs.item_bytes"


def test_parse_request_errors_never_quote_input():
    secret = "the launch code is hunter2"
    with pytest.raises(ProtocolError) as excinfo:
        parse_request(
            {"version": PROTOCOL_VERSION, "inputs": [secret], "outputs": ["dense"]},
            limits=Limits(max_input_bytes=4),
        )
    assert "hunter2" not in str(excinfo.value)


# --------------------------------------------------------------------------
# Sparse canonicalization
# --------------------------------------------------------------------------


def test_canonical_sparse_sorts_and_drops_zeros():
    canonical = canonical_sparse({"8271": 0.37, "1012": 0.91, "5": 0.0}, vocabulary=250002)
    assert canonical == {"indices": [1012, 8271], "values": [0.91, 0.37]}


def test_canonical_sparse_rejects_out_of_vocabulary():
    with pytest.raises(ProtocolError) as excinfo:
        canonical_sparse({"250002": 0.5}, vocabulary=250002)
    assert excinfo.value.invariant == "sparse.range"


def test_canonical_sparse_rejects_non_finite():
    with pytest.raises(ProtocolError) as excinfo:
        canonical_sparse({"1": float("inf")}, vocabulary=10)
    assert excinfo.value.invariant == "sparse.finite"


def test_canonical_sparse_bounds_nonzero_count():
    weights = {str(i): 0.5 for i in range(10)}
    with pytest.raises(ProtocolError) as excinfo:
        canonical_sparse(weights, vocabulary=100, limits=Limits(max_sparse_nonzero=5))
    assert excinfo.value.invariant == "sparse.limit"


# --------------------------------------------------------------------------
# Response building
# --------------------------------------------------------------------------


def dense_space(dimensions: int = 2) -> Space:
    return Space(
        id="space-dense",
        provider="custom",
        model="BAAI/bge-m3",
        kind=DENSE,
        dimensions=dimensions,
        metric="cosine",
        revision="rev",
        tokenizer="tok",
    )


def dense_request(count: int = 1) -> Request:
    return Request(inputs=["a"] * count, outputs=(DENSE,))


def test_build_response_shape(response_schema):
    jsonschema = pytest.importorskip("jsonschema")
    request = dense_request(2)
    payload = build_response(
        "BAAI/bge-m3",
        {DENSE: dense_space()},
        [Item(dense=[0.1, 0.2]), Item(dense=[0.3, 0.4])],
        request,
        Usage(input_tokens=4, request_count=1),
    )
    jsonschema.validate(payload, response_schema)
    assert payload["version"] == PROTOCOL_VERSION
    assert len(payload["data"]) == 2
    assert payload["usage"]["input_bytes"] == 2


def test_build_response_rejects_wrong_cardinality():
    with pytest.raises(ProtocolError) as excinfo:
        build_response("m", {DENSE: dense_space()}, [Item(dense=[0.1, 0.2])], dense_request(2))
    assert excinfo.value.invariant == "data.cardinality"


def test_build_response_rejects_missing_representation():
    with pytest.raises(ProtocolError) as excinfo:
        build_response("m", {DENSE: dense_space()}, [Item()], dense_request())
    assert excinfo.value.invariant == "data.missing"


def test_build_response_rejects_unrequested_representation():
    item = Item(dense=[0.1, 0.2], multi_vector=[[0.1, 0.2]])
    with pytest.raises(ProtocolError) as excinfo:
        build_response("m", {DENSE: dense_space()}, [item], dense_request())
    assert excinfo.value.invariant == "data.unrequested"


def test_build_response_rejects_width_mismatch():
    with pytest.raises(ProtocolError) as excinfo:
        build_response("m", {DENSE: dense_space(3)}, [Item(dense=[0.1, 0.2])], dense_request())
    assert excinfo.value.invariant == "dense.width"


def test_build_response_rejects_non_finite_dense():
    with pytest.raises(ProtocolError) as excinfo:
        build_response("m", {DENSE: dense_space()}, [Item(dense=[0.1, float("nan")])], dense_request())
    assert excinfo.value.invariant == "dense.finite"


def test_build_response_rejects_missing_space():
    with pytest.raises(ProtocolError) as excinfo:
        build_response("m", {}, [Item(dense=[0.1, 0.2])], dense_request())
    assert excinfo.value.invariant == "spaces.missing"


def test_build_response_rejects_invalid_space():
    broken = Space(id="", provider="custom", model="m", kind=DENSE, dimensions=2, metric="cosine")
    with pytest.raises(ProtocolError) as excinfo:
        build_response("m", {DENSE: broken}, [Item(dense=[0.1, 0.2])], dense_request())
    assert excinfo.value.invariant == "space.id"


def test_build_response_rejects_unsorted_sparse():
    space = Space(id="s", provider="custom", model="m", kind=SPARSE, dimensions=100, metric="dot_product")
    item = Item(sparse={"indices": [5, 2], "values": [0.1, 0.2]})
    request = Request(inputs=["a"], outputs=(SPARSE,))
    with pytest.raises(ProtocolError) as excinfo:
        build_response("m", {SPARSE: space}, [item], request)
    assert excinfo.value.invariant == "sparse.order"


def test_build_response_rejects_ragged_multi_vector():
    space = Space(id="mv", provider="custom", model="m", kind=MULTI_VECTOR, dimensions=2, metric="cosine")
    item = Item(multi_vector=[[0.1, 0.2], [0.3]])
    request = Request(inputs=["a"], outputs=(MULTI_VECTOR,))
    with pytest.raises(ProtocolError) as excinfo:
        build_response("m", {MULTI_VECTOR: space}, [item], request)
    assert excinfo.value.invariant == "multi_vector.width"


# --------------------------------------------------------------------------
# Handler
# --------------------------------------------------------------------------


class FakeModel:
    """Stands in for BGEM3FlagModel, recording how often it was asked to work.

    Outputs are derived from input length so that batch order is checkable.
    """

    def __init__(self) -> None:
        self.calls: List[Mapping[str, Any]] = []

    def encode(self, texts, return_dense=False, return_sparse=False, return_colbert_vecs=False):
        self.calls.append(
            {
                "texts": list(texts),
                "dense": return_dense,
                "sparse": return_sparse,
                "colbert": return_colbert_vecs,
            }
        )
        out: Dict[str, Any] = {}
        if return_dense:
            out["dense_vecs"] = [[float(len(text)), 0.5] for text in texts]
        if return_sparse:
            out["lexical_weights"] = [{str(len(text) % 250002): 0.75} for text in texts]
        if return_colbert_vecs:
            out["colbert_vecs"] = [[[float(len(text)), 0.25]] for text in texts]
        return out


def handler_config(outputs=("dense", "sparse", "multi_vector")) -> Config:
    return Config(
        model_name="BAAI/bge-m3",
        revision="immutable-revision",
        tokenizer_revision="immutable-tokenizer-revision",
        dense_dimensions=2,
        sparse_vocabulary=250002,
        outputs=tuple(outputs),
        space_ids={kind: f"space-{kind}" for kind in outputs},
    )


@pytest.fixture()
def handler() -> EndpointHandler:
    return EndpointHandler(model=FakeModel(), config=handler_config())


def test_handler_returns_one_item_per_input_in_order(handler, response_schema):
    jsonschema = pytest.importorskip("jsonschema")
    payload = handler(
        {
            "version": PROTOCOL_VERSION,
            "inputs": ["a", "bb", "ccc"],
            "input_type": "document",
            "outputs": ["dense", "sparse"],
        }
    )
    jsonschema.validate(payload, response_schema)
    assert len(payload["data"]) == 3
    assert [item["dense"][0] for item in payload["data"]] == [1.0, 2.0, 3.0]
    assert payload["spaces"]["sparse"]["metric"] == "dot_product"
    assert payload["spaces"]["dense"]["id"] == "space-dense"


def test_handler_returns_only_requested_outputs(handler):
    payload = handler({"version": PROTOCOL_VERSION, "inputs": ["a"], "outputs": ["dense"]})
    item = payload["data"][0]
    assert "dense" in item
    assert "sparse" not in item and "multi_vector" not in item
    assert set(payload["spaces"]) == {"dense"}
    # Costly kinds must not be computed either, not merely dropped afterwards.
    assert handler.model.calls[0] == {
        "texts": ["a"],
        "dense": True,
        "sparse": False,
        "colbert": False,
    }


def test_handler_multi_vector_is_opt_in(handler):
    payload = handler(
        {"version": PROTOCOL_VERSION, "inputs": ["hello"], "outputs": ["multi_vector"]}
    )
    assert payload["data"][0]["multi_vector"] == [[5.0, 0.25]]
    assert handler.model.calls[0]["colbert"] is True


def test_handler_tokenizes_the_batch_in_one_call(handler):
    handler({"version": PROTOCOL_VERSION, "inputs": ["a", "bb"], "outputs": ["dense"]})
    assert len(handler.model.calls) == 1
    assert handler.model.calls[0]["texts"] == ["a", "bb"]


def test_handler_rejects_unsupported_output():
    dense_only = EndpointHandler(model=FakeModel(), config=handler_config(outputs=("dense",)))
    with pytest.raises(ProtocolError) as excinfo:
        dense_only({"version": PROTOCOL_VERSION, "inputs": ["a"], "outputs": ["sparse"]})
    assert excinfo.value.invariant == "outputs.unsupported"


def test_handler_health_check_does_no_inference_and_leaks_nothing(handler):
    for probe in ({}, {"health": True}):
        payload = handler(probe)
        assert payload == {"status": "ok", "version": PROTOCOL_VERSION}
    assert handler.model.calls == []


def test_handler_loads_the_model_once_per_process():
    model = FakeModel()
    handler = EndpointHandler(model=model, config=handler_config())
    for _ in range(3):
        handler({"version": PROTOCOL_VERSION, "inputs": ["a"], "outputs": ["dense"]})
    assert handler.model is model
    assert len(model.calls) == 3


def test_config_requires_declared_space_ids(monkeypatch):
    monkeypatch.setenv("AGENTIC_OUTPUTS", "dense")
    monkeypatch.delenv("AGENTIC_SPACE_ID_DENSE", raising=False)
    with pytest.raises(ProtocolError) as excinfo:
        Config.from_env()
    assert excinfo.value.invariant == "config.space_id"


def test_config_reads_immutable_revisions(monkeypatch):
    monkeypatch.setenv("AGENTIC_OUTPUTS", "dense,sparse")
    monkeypatch.setenv("AGENTIC_SPACE_ID_DENSE", "d1")
    monkeypatch.setenv("AGENTIC_SPACE_ID_SPARSE", "s1")
    monkeypatch.setenv("AGENTIC_MODEL_REVISION", "abc123")
    monkeypatch.setenv("AGENTIC_TOKENIZER_REVISION", "tok123")

    config = Config.from_env()
    assert config.space("dense").revision == "abc123"
    assert config.space("sparse").tokenizer == "tok123"
    assert config.space("sparse").dimensions == 250002


def test_config_rejects_unknown_output(monkeypatch):
    monkeypatch.setenv("AGENTIC_OUTPUTS", "colbert")
    with pytest.raises(ProtocolError) as excinfo:
        Config.from_env()
    assert excinfo.value.invariant == "config.outputs"


# --------------------------------------------------------------------------
# SageMaker entrypoint
# --------------------------------------------------------------------------


def test_sagemaker_entrypoint_round_trip():
    import sagemaker_entrypoint as entry

    handler = EndpointHandler(model=FakeModel(), config=handler_config())
    body = json.dumps({"version": PROTOCOL_VERSION, "inputs": ["a"], "outputs": ["dense"]})

    data = entry.input_fn(body.encode("utf-8"), "application/json")
    prediction = entry.predict_fn(data, handler)
    payload, content_type = entry.output_fn(prediction, "application/json")

    assert content_type == "application/json"
    assert json.loads(payload)["data"][0]["dense"] == [1.0, 0.5]


def test_sagemaker_entrypoint_rejects_other_content_types():
    import sagemaker_entrypoint as entry

    with pytest.raises(ProtocolError):
        entry.input_fn("{}", "text/csv")
    with pytest.raises(ProtocolError):
        entry.output_fn({}, "text/csv")
    # A client that states no preference gets JSON.
    assert entry.output_fn({}, "*/*")[1] == "application/json"
