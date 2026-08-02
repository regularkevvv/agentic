// Example: learned sparse encoding with the model running in this process.
//
// Every other example here needs an API key. This one needs none, because
// nothing leaves the machine: the model is a file, ONNX Runtime executes it,
// and the vectors are built in Go. That is the whole difference, and it is
// worth seeing next to the hosted providers rather than described.
//
// What it shows is term expansion. A SPLADE-family model scores the entire
// vocabulary through a masked-language-model head, so a document about an
// "automobile" comes back carrying weight on "car" and "motor" — words it never
// contained. That is what lets lexical retrieval survive a vocabulary mismatch,
// and it is why the coordinates below are worth reading rather than counting.
//
// Producing the model file is a one-time step:
//
//	uv run provider/local/onnx/export_onnx.py --out ./models
//
// Then point this at it:
//
//	AGENTIC_ONNX_MODEL=./models/granite-embedding-30m-sparse.onnx \
//	AGENTIC_ONNX_TOKENIZER=~/.cache/huggingface/hub/models--ibm-granite--granite-embedding-30m-sparse/snapshots/<rev>/tokenizer.json \
//	AGENTIC_ONNX_LIBRARY=/path/to/libonnxruntime.dylib \
//	go run ./examples/sparse
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sort"
	"text/tabwriter"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/local/onnx"
)

// documents are ordinary sentences that share no vocabulary with the query
// below. Exact-match lexical search scores every one of them zero; an expanding
// model does not, which is the point being demonstrated.
var documents = []string{
	"The automobile industry shifted to electric drivetrains.",
	"She repaired the engine of a vintage roadster.",
	"Sourdough needs a long, cold fermentation.",
	"The quarterly report understated depreciation.",
}

const query = "car"

func main() {
	model := os.Getenv("AGENTIC_ONNX_MODEL")
	tokenizer := os.Getenv("AGENTIC_ONNX_TOKENIZER")
	if model == "" || tokenizer == "" {
		log.Fatal("set AGENTIC_ONNX_MODEL and AGENTIC_ONNX_TOKENIZER; " +
			"see the comment at the top of this file for how to produce them")
	}

	// The space is declared by the caller, never derived from the file. An
	// endpoint cannot prove its own weights revision, and neither can a graph on
	// disk: if the identity an index is keyed on were computed at runtime, a
	// swapped model would produce vectors that land in the same space as the old
	// ones and retrieve slightly worse forever, with nothing failing.
	encoder, err := onnx.New(model, tokenizer, agentic.VectorSpace{
		Provider: "onnx",
		Model:    "ibm-granite/granite-embedding-30m-sparse",
		Kind:     agentic.RepresentationSparse,
		Metric:   agentic.SimilarityDotProduct,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := encoder.Close(); err != nil {
			log.Print(err)
		}
	}()

	ctx := context.Background()
	docs, err := agentic.EncodeDocuments(ctx, encoder, documents, agentic.RepresentationSparse)
	if err != nil {
		log.Fatal(err)
	}
	queries, err := agentic.EncodeQueries(ctx, encoder, []string{query}, agentic.RepresentationSparse)
	if err != nil {
		log.Fatal(err)
	}

	expansion(docs.Data[0].Sparse, docs.Spaces[agentic.RepresentationSparse])
	ranking(queries.Data[0].Sparse, docs)
}

// expansion prints how much of the first document's vector is words the
// document does not contain.
func expansion(vector *agentic.SparseVector, space agentic.VectorSpace) {
	fmt.Printf("\nEncoded in this process. No network, no API key.\n")
	fmt.Printf("  model       %s\n", space.Model)
	fmt.Printf("  vocabulary  %d slots\n", space.Dimensions)
	fmt.Printf("  document    %q\n", documents[0])
	fmt.Printf("  weighted    %d slots — %.2f%% of the vocabulary\n\n",
		vector.Len(), 100*float64(vector.Len())/float64(space.Dimensions))

	// Sorting by weight rather than index: the coordinate order is the storage
	// order, and says nothing about which terms the model considered important.
	type coordinate struct {
		index  uint32
		weight float32
	}
	ranked := make([]coordinate, vector.Len())
	for i := range ranked {
		ranked[i] = coordinate{vector.Indices[i], vector.Values[i]}
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].weight > ranked[j].weight })

	out := table()
	row(out, "slot", "weight")
	for _, c := range ranked[:min(10, len(ranked))] {
		row(out, fmt.Sprint(c.index), fmt.Sprintf("%.4f", c.weight))
	}
	flush(out)
	fmt.Printf("\n  Coordinates are vocabulary slots, not words. Which words they\n" +
		"  stand for is a property of the tokenizer, so an index keyed on\n" +
		"  strings is keyed on the wrong thing.\n")
}

// ranking scores the query against every document, which is where expansion
// stops being a curiosity.
func ranking(encoded *agentic.SparseVector, docs *agentic.RepresentationResponse) {
	fmt.Printf("\nQuery %q against documents that never contain it:\n\n", query)

	out := table()
	row(out, "score", "document")
	type scored struct {
		score float64
		text  string
	}
	results := make([]scored, len(documents))
	for i, doc := range docs.Data {
		results[i] = scored{dot(encoded, doc.Sparse), documents[i]}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].score > results[j].score })
	for _, r := range results {
		row(out, fmt.Sprintf("%.4f", r.score), r.text)
	}
	flush(out)
	fmt.Printf("\n  A model that only weighted words it was given would score every\n" +
		"  one of these zero.\n\n")
}

// dot is the similarity for a dot-product space: sum the weights of the
// coordinates the two vectors share. Both are sorted, so one pass suffices.
func dot(query, document *agentic.SparseVector) float64 {
	var total float64
	for i, j := 0, 0; i < query.Len() && j < document.Len(); {
		switch {
		case query.Indices[i] == document.Indices[j]:
			total += float64(query.Values[i]) * float64(document.Values[j])
			i++
			j++
		case query.Indices[i] < document.Indices[j]:
			i++
		default:
			j++
		}
	}
	return total
}

// Writes to a tabwriter cannot meaningfully fail on stdout, and checking each
// one would bury the output this example exists to show. They are discarded in
// one place instead, which is what the other examples here do.
func table() *tabwriter.Writer {
	return tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
}

func row(w *tabwriter.Writer, left, right string) {
	_, _ = fmt.Fprintf(w, "  %s\t%s\n", left, right)
}

func flush(w *tabwriter.Writer) { _ = w.Flush() }
