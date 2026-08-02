# /// script
# requires-python = "==3.11.*"
# dependencies = ["torch==2.13.0", "transformers", "onnx", "onnxscript"]
# ///
"""Export a SPLADE-family sparse model to ONNX, with a PyTorch reference beside it.

This is the only Python in the repository, and it runs once per model. The
package beside it runs the exported graph in Go and never invokes Python: there
is no runtime dependency here, only a build-time one, in the sense that a
compiler is a build-time dependency.

    uv run export_onnx.py --out /some/writable/directory

The header above is PEP 723 inline metadata. uv reads it, fetches Python 3.11 if
the machine has none, builds a throwaway environment, runs this, and discards
it. Nothing is installed system-wide and there is no virtualenv to manage.

Without uv, `pip install torch==2.13.0 transformers onnx onnxscript` and
`python export_onnx.py` do the same thing.

Two artifacts come out. `<model>.onnx` is the graph, about 117 MiB in fp32 for
granite-embedding-30m-sparse, and belongs nowhere near version control.
`reference.json` is small, is checked in as
`provider/local/onnx/testdata/granite_reference.json`, and is what makes the Go
implementation falsifiable: PyTorch's own input ids, nonzero count, top terms,
and every coordinate for each sample input.

Regenerating the reference on a different `transformers` release reproduces
every coordinate index and moves the weights by around 4e-06 — the same order
as the PyTorch-to-ONNX difference, and far inside the 1e-4 the Go test gates
on. Measured 2026-08-01.

The graph is the raw masked-language-model — (input_ids, attention_mask) ->
logits — with both the batch and sequence axes dynamic. Pooling is deliberately
left outside it so the Go side has to implement the same log1p(relu(x)) * mask
-> max reduction, which is the part worth proving. Baking it in would export the
disagreement rather than expose it.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Sequence

import onnx
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# The default target. It expands — a document about an "automobile" acquires
# weight on "car" — it is thirty million parameters, and it loads as a stock
# transformers architecture with no custom modelling code.
DEFAULT_MODEL = "ibm-granite/granite-embedding-30m-sparse"

# One short input and one long one, so the reference exercises both a single
# wordpiece and a sentence, and so a batch of the two has padding in it. The
# coined term "quensel" is there because no tokenizer has it as a single token:
# it forces the subword path that a real rare-term query takes.
DEFAULT_TEXTS = (
    "automobile",
    "The quensel actuator must be recalibrated before every third launch.",
)

# The exported sequence bound. It is the model's positional limit, and
# provider/local/onnx rejects longer inputs rather than truncating them.
MAX_SEQUENCE = 512


def pool(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The SPLADE reduction, restated here rather than imported.

    handler_splade.py performs the same four lines. Computing the reference
    through a second implementation is the point: a shared helper would make an
    error in it invisible to both sides.
    """
    weighted = torch.log1p(torch.relu(logits))
    return (weighted * mask.unsqueeze(-1)).max(dim=1).values


def export(model, tokenizer, path: str) -> None:
    """Write the ONNX graph, with dynamic batch and sequence axes."""
    # Two rows rather than one, as insurance. torch.export specializes axes
    # whose example extent is 0 or 1 unless a constraint says otherwise; the
    # explicit Dim below is that constraint, and measured on torch 2.13 a
    # one-row sample does keep the batch axis symbolic. The second row costs
    # nothing and removes the dependence on that behaviour holding.
    #
    # verify_dynamic is the part that matters, because specialization is silent
    # here and surfaces much later inside ONNX Runtime as "Got invalid
    # dimensions for input: input_ids" — at which point the graph is already
    # deployed.
    sample = tokenizer(["hello world", "a second row, so batch stays dynamic"],
                       padding=True, return_tensors="pt")
    batch = torch.export.Dim("batch", min=1, max=64)
    sequence = torch.export.Dim("seq", min=2, max=MAX_SEQUENCE)

    program = torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        dynamo=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes={
            "input_ids": {0: batch, 1: sequence},
            "attention_mask": {0: batch, 1: sequence},
        },
    )
    program.save(path)
    verify_dynamic(path)


def verify_dynamic(path: str) -> None:
    """Fail unless the saved graph's batch and sequence axes are symbolic.

    The axis names torch chooses vary between releases and do not matter; that
    they are names rather than fixed extents does. A graph with a literal 1 in
    either position encodes one input at a time, which no consumer of this
    export wants and none of them can detect until they try to batch.
    """
    graph = onnx.load(path, load_external_data=False).graph
    for value in graph.input:
        dims = value.type.tensor_type.shape.dim
        for axis, name in ((0, "batch"), (1, "sequence")):
            if not dims[axis].dim_param:
                raise SystemExit(
                    f"{value.name} has a fixed {name} axis of {dims[axis].dim_value}; "
                    "the export specialized it and the graph is unusable"
                )


def reference_for(model, tokenizer, text: str) -> Dict[str, Any]:
    """Encode one input in PyTorch and record everything Go has to reproduce."""
    encoded = tokenizer([text], return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits
    pooled = pool(logits, encoded["attention_mask"])[0]

    nonzero = torch.nonzero(pooled).flatten().tolist()
    # Six decimals is far finer than the 1e-4 the Go test gates on, and keeps a
    # file with tens of thousands of coordinates diffable by a human.
    coordinates = [[int(i), round(float(pooled[i]), 6)] for i in nonzero]
    coordinates.sort(key=lambda pair: -pair[1])

    return {
        "text": text,
        "input_ids": encoded["input_ids"][0].tolist(),
        "nonzero_count": len(coordinates),
        "top": [
            {"index": index, "weight": weight, "term": tokenizer.convert_ids_to_tokens(index)}
            for index, weight in coordinates[:20]
        ],
        # Sorted by index, which is the order a SparseVector's coordinates are
        # in, so the Go test compares position by position without re-sorting.
        "all_coordinates": sorted(coordinates),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model id or local path")
    parser.add_argument("--out", default=".", help="directory for the artifacts")
    parser.add_argument(
        "--text",
        action="append",
        dest="texts",
        help="sample input for the reference; repeatable, defaults to two built-in inputs",
    )
    args = parser.parse_args(argv)
    texts: List[str] = args.texts or list(DEFAULT_TEXTS)

    os.makedirs(args.out, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # .float() rather than a from_pretrained dtype argument: the keyword for it
    # has been spelled both torch_dtype and dtype across transformers releases,
    # and this says the same thing in every one of them. fp32 is not optional —
    # a half-precision export would disagree with this reference by far more
    # than the 1e-4 the Go test gates on.
    model = AutoModelForMaskedLM.from_pretrained(args.model).float().eval()

    graph_path = os.path.join(args.out, args.model.split("/")[-1] + ".onnx")
    print(f"exporting {args.model}...", flush=True)
    export(model, tokenizer, graph_path)
    print(f"wrote {graph_path} ({os.path.getsize(graph_path) / 1e6:.1f} MB)", flush=True)

    inputs = []
    for text in texts:
        entry = reference_for(model, tokenizer, text)
        inputs.append(entry)
        top = ", ".join(term["term"] for term in entry["top"][:5])
        print(f'"{text}" -> {entry["nonzero_count"]} coordinates, top: {top}', flush=True)

    reference_path = os.path.join(args.out, "reference.json")
    with open(reference_path, "w") as fh:
        json.dump(
            {
                "model": args.model,
                # The width of the sparse space. provider/local/onnx reads the same
                # number out of the graph's output shape rather than trusting
                # this, and the live test checks the two agree.
                "vocabulary": model.config.vocab_size,
                "inputs": inputs,
            },
            fh,
            indent=2,
        )
    print(f"wrote {reference_path}", flush=True)
    print(
        "\nCopy reference.json to provider/local/onnx/testdata/granite_reference.json "
        "and point AGENTIC_ONNX_MODEL at the graph.",
        flush=True,
    )


if __name__ == "__main__":
    main()
