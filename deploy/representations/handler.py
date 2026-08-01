"""Custom inference handler speaking agentic.representations.v1.

Deployed to a Hugging Face Inference Endpoint as-is, and to SageMaker through
sagemaker_entrypoint.py. Both wrap the same class, because the transport
differs between them and the payload does not.

The model is loaded once per process, in ``__init__``. Loading it per request
would add seconds of latency to every call and, on a shared GPU, eventually
run the endpoint out of memory.

Configuration is immutable and comes from the environment. In particular the
vector-space IDs are declared by the operator, not derived here: an endpoint
cannot prove its own weights revision, and an identity a consumer keys an
index on must not be something the runtime can quietly change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from protocol import (
    DENSE,
    KINDS,
    METRIC_COSINE,
    METRIC_DOT_PRODUCT,
    MULTI_VECTOR,
    PROTOCOL_VERSION,
    SPARSE,
    Item,
    Limits,
    ProtocolError,
    Space,
    Usage,
    build_response,
    canonical_sparse,
    parse_request,
)


@dataclass(frozen=True)
class Config:
    """Immutable deployment configuration."""

    model_name: str
    revision: str
    tokenizer_revision: str
    dense_dimensions: int
    sparse_vocabulary: int
    outputs: Tuple[str, ...]
    space_ids: Mapping[str, str]
    provider: str = "custom"
    use_fp16: bool = True

    @staticmethod
    def from_env(model_path: str = "") -> "Config":
        model_name = os.environ.get("AGENTIC_MODEL") or model_path or "BAAI/bge-m3"
        outputs = tuple(
            kind.strip()
            for kind in os.environ.get("AGENTIC_OUTPUTS", ",".join(KINDS)).split(",")
            if kind.strip()
        )
        for kind in outputs:
            if kind not in KINDS:
                raise ProtocolError("config.outputs", f"unknown representation kind {kind!r}")

        space_ids = {}
        for kind in outputs:
            env = f"AGENTIC_SPACE_ID_{kind.upper()}"
            value = os.environ.get(env, "")
            if not value:
                raise ProtocolError(
                    "config.space_id",
                    f"{env} must be set; a vector space identity is declared by the "
                    "operator, not derived at runtime",
                )
            space_ids[kind] = value

        return Config(
            model_name=model_name,
            revision=os.environ.get("AGENTIC_MODEL_REVISION", ""),
            tokenizer_revision=os.environ.get("AGENTIC_TOKENIZER_REVISION", ""),
            dense_dimensions=int(os.environ.get("AGENTIC_DENSE_DIMENSIONS", "1024")),
            sparse_vocabulary=int(os.environ.get("AGENTIC_SPARSE_VOCABULARY", "250002")),
            outputs=outputs,
            space_ids=space_ids,
            provider=os.environ.get("AGENTIC_PROVIDER", "custom"),
            use_fp16=os.environ.get("AGENTIC_FP16", "1") != "0",
        )

    def space(self, kind: str) -> Space:
        dimensions = self.dense_dimensions
        metric = METRIC_COSINE
        if kind == SPARSE:
            dimensions = self.sparse_vocabulary
            metric = METRIC_DOT_PRODUCT
        return Space(
            id=self.space_ids[kind],
            provider=self.provider,
            model=self.model_name,
            revision=self.revision,
            tokenizer=self.tokenizer_revision,
            kind=kind,
            dimensions=dimensions,
            metric=metric,
        )


def load_model(config: Config):
    """Load BGE-M3 once. Imported lazily so protocol tests need no ML stack."""
    from FlagEmbedding import BGEM3FlagModel  # noqa: PLC0415

    return BGEM3FlagModel(
        config.model_name,
        use_fp16=config.use_fp16,
        normalize_embeddings=True,
    )


class EndpointHandler:
    """The entry point Hugging Face Inference Endpoints calls."""

    def __init__(self, path: str = "", model: Any = None, config: Optional[Config] = None) -> None:
        self.config = config or Config.from_env(path)
        self.limits = Limits()
        # Injectable so a contract test can run the protocol path without a
        # GPU, and so SageMaker can reuse an already-loaded model.
        self.model = model if model is not None else load_model(self.config)

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        if _is_health_check(data):
            # No inference, and no configuration echoed back: a health probe is
            # usually reachable from more places than the inference route is.
            return {"status": "ok", "version": PROTOCOL_VERSION}

        request = parse_request(data, supported=self.config.outputs, limits=self.limits)

        encoded = self.model.encode(
            list(request.inputs),
            return_dense=request.wants(DENSE),
            return_sparse=request.wants(SPARSE),
            return_colbert_vecs=request.wants(MULTI_VECTOR),
        )

        items = [
            self._item(encoded, position, request.wants(DENSE), request.wants(SPARSE),
                       request.wants(MULTI_VECTOR))
            for position in range(len(request.inputs))
        ]

        spaces = {kind: self.config.space(kind) for kind in request.outputs}
        usage = Usage(request_count=1)
        return build_response(self.config.model_name, spaces, items, request, usage)

    def _item(self, encoded: Mapping[str, Any], position: int,
              dense: bool, sparse: bool, multi_vector: bool) -> Item:
        item = Item()
        if dense:
            item.dense = [float(value) for value in encoded["dense_vecs"][position]]
        if sparse:
            item.sparse = canonical_sparse(
                encoded["lexical_weights"][position],
                self.config.sparse_vocabulary,
                self.limits,
            )
        if multi_vector:
            item.multi_vector = [
                [float(value) for value in vector]
                for vector in encoded["colbert_vecs"][position]
            ]
        return item


def _is_health_check(data: Mapping[str, Any]) -> bool:
    """A probe is anything that asks for no work: an empty body, or health."""
    if not isinstance(data, Mapping):
        return False
    if data.get("health") is True:
        return True
    return not data
