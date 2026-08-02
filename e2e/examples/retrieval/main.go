// Example: hybrid retrieval with dense and learned sparse representations.
//
// It encodes a small corpus with BGE-M3, which produces both kinds from one
// forward pass, then scores two queries under each to show what the two
// representations are actually good at.
//
// Dense ranks by meaning: the paraphrase query shares no words at all with the
// document that answers it, and dense still puts it first.
//
// Sparse ranks by term overlap, and its useful property is what it does to
// non-matches — it scores them at exactly zero. A nonzero sparse score is
// evidence of real lexical overlap, where a middling dense score is only
// evidence of vague topical similarity. Watch the coined term "quensel": both
// representations rank it first, but sparse separates it from everything else
// completely, while dense leaves a wrong document within striking distance.
//
// That is why hybrid retrieval combines them, and why neither replaces the
// other.
//
// The scoring is deliberately naive — a few lines of arithmetic over slices.
// That is the point of the boundary: Agentic hands you the vectors and the
// metric to compare them under, and your index does the real work. Nothing in
// Agentic returns a score, because only your store knows how it ranks.
//
// Run with DEEPINFRA_TOKEN set, or a .env file beside this program.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"text/tabwriter"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/e2e/examples/internal/envutil"
	"github.com/regularkevvv/agentic/e2e/internal/corpus"
	"github.com/regularkevvv/agentic/provider/deepinfra"
)

// The queries, the corpus, and the two scoring functions come from
// internal/corpus because the live provider tests assert exactly the ranking
// this program prints. A local copy would let the demonstration drift away
// from what is actually verified.
var queries = []string{corpus.ParaphraseQuery, corpus.RareTermQuery}

// storedDocument is what an application actually persists. The space IDs
// travel with the vectors because vectors alone cannot be checked for
// compatibility later, and querying an index with a differently-encoded vector
// fails silently — as slightly worse recall, never as an error.
type storedDocument struct {
	Text          string
	Dense         []float32
	Sparse        *agentic.SparseVector
	DenseSpaceID  string
	SparseSpaceID string
}

func main() {
	if err := envutil.LoadDotEnv(); err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	encoder, err := deepinfra.New(deepinfra.BGEM3Model)
	if err != nil {
		log.Fatal(err)
	}

	// One request, both representations. BGE-M3 computes them in a single
	// forward pass, so asking for them separately would pay twice.
	docs, err := agentic.EncodeDocuments(ctx, encoder, corpus.Documents,
		agentic.RepresentationDense,
		agentic.RepresentationSparse,
	)
	if err != nil {
		log.Fatal(err)
	}

	denseSpace := docs.Spaces[agentic.RepresentationDense]
	sparseSpace := docs.Spaces[agentic.RepresentationSparse]

	fmt.Println("vector spaces (store these beside the vectors):")
	fmt.Printf("  dense   %s\n", denseSpace)
	fmt.Printf("  sparse  %s\n", sparseSpace)
	fmt.Printf("\nencoded %d documents in %d request(s), %d input tokens\n\n",
		len(docs.Data), docs.Usage.RequestCount, docs.Usage.InputTokens)

	index := make([]storedDocument, len(corpus.Documents))
	for i, item := range docs.Data {
		index[i] = storedDocument{
			Text:          corpus.Documents[i],
			Dense:         item.Dense,
			Sparse:        item.Sparse,
			DenseSpaceID:  denseSpace.ID,
			SparseSpaceID: sparseSpace.ID,
		}
	}

	encoded, err := agentic.EncodeQueries(ctx, encoder, queries,
		agentic.RepresentationDense,
		agentic.RepresentationSparse,
	)
	if err != nil {
		log.Fatal(err)
	}

	// Refuse to compare vectors from different spaces. This is the check the
	// space descriptor exists for, and the one an application must not skip.
	querySpace := encoded.Spaces[agentic.RepresentationDense]
	if querySpace.ID != denseSpace.ID {
		log.Fatalf("query space %s is not the index space %s", querySpace.ID, denseSpace.ID)
	}

	for i, query := range queries {
		fmt.Printf("query: %q\n", query)
		report(index, encoded.Data[i])
		fmt.Println()
	}

	fmt.Println("Dense ranks the paraphrase first even though it shares no words with the query.")
	fmt.Println("Sparse scores every document with no term overlap at exactly zero, so a")
	fmt.Println("nonzero sparse score means something a middling dense score does not.")
	fmt.Println()
	fmt.Println("Combining the two rankings is your retrieval system's job, not Agentic's:")
	fmt.Println("nothing here returns a score, because only your index knows how it ranks.")
}

// report prints both rankings side by side.
func report(index []storedDocument, query agentic.Representation) {
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	_, _ = fmt.Fprintln(w, "  dense\tsparse\tdocument")

	bestDense, bestSparse := 0, 0
	denseScores := make([]float64, len(index))
	sparseScores := make([]float64, len(index))

	for i, doc := range index {
		denseScores[i] = corpus.Cosine(query.Dense, doc.Dense)
		sparseScores[i] = corpus.DotProduct(query.Sparse, doc.Sparse)
		if denseScores[i] > denseScores[bestDense] {
			bestDense = i
		}
		if sparseScores[i] > sparseScores[bestSparse] {
			bestSparse = i
		}
	}

	for i, doc := range index {
		marker := "  "
		switch {
		case i == bestDense && i == bestSparse:
			marker = "**"
		case i == bestDense:
			marker = "d "
		case i == bestSparse:
			marker = "s "
		}
		_, _ = fmt.Fprintf(w, "%s%.4f\t%.4f\t%s\n", marker, denseScores[i], sparseScores[i], truncate(doc.Text, 60))
	}
	_ = w.Flush()
}

func truncate(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	return s[:limit-1] + "…"
}
