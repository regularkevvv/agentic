//go:build e2e
// +build e2e

// This file exercises the multi-representation encoders against live APIs.
//
// The transport tests prove that each provider sends what we believe its API
// wants and decodes what we believe it returns. Only these tests prove that
// the resulting vectors retrieve: that a dense paraphrase match, a sparse rare
// term match, and a late-interaction match each put the right document first.
// Asserting HTTP 200 would prove none of that.
//
// Each provider skips cleanly when its gate is absent:
//   - DEEPINFRA_TOKEN                      DeepInfra native BGE-M3
//
// The corpus is deliberately tiny. These are metered APIs, and a handful of
// short documents is enough to separate a working encoder from a broken one.
package e2e

import (
	"context"
	"math"
	"os"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/deepinfra"
)

// representationDocs has one document per retrieval mode being tested.
//
// budgetDoc is the paraphrase target: it answers the query without sharing its
// wording, so only a dense match finds it. quenselDoc carries a coined term
// that appears in no other document and in no vocabulary, so only a lexical
// match finds it.
var representationDocs = []string{
	"Sourdough bread is made by fermenting dough with naturally occurring lactobacilli.",
	"Engineering spends two thousand dollars a month running its infrastructure.",
	"The quensel actuator must be recalibrated before every third launch.",
	"Mount Everest is the highest mountain above sea level.",
}

const (
	budgetDoc  = 1
	quenselDoc = 2

	paraphraseQuery = "how much does the engineering team cost each month?"
	rareTermQuery   = "quensel actuator recalibration"
)

func skipIfNoDeepInfraToken(t *testing.T) {
	t.Helper()
	if os.Getenv("DEEPINFRA_TOKEN") == "" {
		t.Skip("DEEPINFRA_TOKEN not set, skipping DeepInfra e2e test")
	}
}

func representationCtx(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// dotProduct scores a sparse query against a sparse document by summing the
// weights of the coordinates they share. This is what a sparse index computes;
// doing it here proves the returned coordinates line up without needing one.
func dotProduct(query, document *agentic.SparseVector) float64 {
	var score float64
	i, j := 0, 0
	for i < len(query.Indices) && j < len(document.Indices) {
		switch {
		case query.Indices[i] < document.Indices[j]:
			i++
		case query.Indices[i] > document.Indices[j]:
			j++
		default:
			score += float64(query.Values[i]) * float64(document.Values[j])
			i++
			j++
		}
	}
	return score
}

// maxSim is ColBERT's late-interaction score: for each query token take its
// best-matching document token, then sum. It is the reason multi-vector output
// must never be averaged into one vector.
func maxSim(query, document [][]float32) float64 {
	var total float64
	for _, q := range query {
		best := math.Inf(-1)
		for _, d := range document {
			if score := cosine(q, d); score > best {
				best = score
			}
		}
		if !math.IsInf(best, -1) {
			total += best
		}
	}
	return total
}

// bestBy returns the index of the document scoring highest under score.
func bestBy(t *testing.T, count int, score func(i int) float64) int {
	t.Helper()
	best, bestScore := -1, math.Inf(-1)
	for i := range count {
		s := score(i)
		t.Logf("document [%d] scores %.4f", i, s)
		if s > bestScore {
			best, bestScore = i, s
		}
	}
	return best
}

func TestE2E_Representations_DeepInfraDenseAndSparse(t *testing.T) {
	skipIfNoDeepInfraToken(t)
	ctx := representationCtx(t)

	encoder, err := deepinfra.New(deepinfra.BGEM3Model)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// One request for both kinds. That it is one request is the point: BGE-M3
	// computes them in a single forward pass, and issuing two calls would pay
	// twice for the same work.
	docs, err := agentic.EncodeDocuments(ctx, encoder, representationDocs,
		agentic.RepresentationDense, agentic.RepresentationSparse)
	if err != nil {
		t.Fatalf("EncodeDocuments: %v", err)
	}
	if len(docs.Data) != len(representationDocs) {
		t.Fatalf("got %d representations for %d documents", len(docs.Data), len(representationDocs))
	}
	if docs.Usage.RequestCount != 1 {
		t.Errorf("request count = %d, want one call for both kinds", docs.Usage.RequestCount)
	}
	if docs.Usage.InputTokens < 0 || docs.Usage.InputBytes < 0 {
		t.Errorf("usage has a negative measurement: %+v", docs.Usage)
	}

	denseSpace, ok := docs.Space(agentic.RepresentationDense)
	if !ok {
		t.Fatal("response describes no dense space")
	}
	sparseSpace, ok := docs.Space(agentic.RepresentationSparse)
	if !ok {
		t.Fatal("response describes no sparse space")
	}
	t.Logf("dense space %s", denseSpace)
	t.Logf("sparse space %s", sparseSpace)

	if denseSpace.Dimensions != 1024 {
		t.Errorf("dense width = %d; BGE-M3 is documented at 1024", denseSpace.Dimensions)
	}
	if sparseSpace.Metric != agentic.SimilarityDotProduct {
		t.Errorf("sparse metric = %q, want dot_product", sparseSpace.Metric)
	}
	if denseSpace.ID == sparseSpace.ID {
		t.Error("dense and sparse share a space ID; they cannot share an index")
	}

	// Every document must carry both kinds, or a partial batch has been
	// accepted somewhere.
	for i, item := range docs.Data {
		if len(item.Dense) != denseSpace.Dimensions {
			t.Fatalf("document %d dense width = %d", i, len(item.Dense))
		}
		if item.Sparse.Len() == 0 {
			t.Fatalf("document %d has no sparse coordinates", i)
		}
	}

	queries, err := agentic.EncodeQueries(ctx, encoder, []string{paraphraseQuery, rareTermQuery},
		agentic.RepresentationDense, agentic.RepresentationSparse)
	if err != nil {
		t.Fatalf("EncodeQueries: %v", err)
	}

	// BGE-M3 is symmetric: the query and document roles encode into the same
	// space, so the two sides are directly comparable.
	querySpace, _ := queries.Space(agentic.RepresentationDense)
	if querySpace.ID != denseSpace.ID {
		t.Fatalf("query space %s differs from document space %s", querySpace.ID, denseSpace.ID)
	}

	t.Run("dense finds the paraphrase", func(t *testing.T) {
		best := bestBy(t, len(docs.Data), func(i int) float64 {
			return cosine(queries.Data[0].Dense, docs.Data[i].Dense)
		})
		if best != budgetDoc {
			t.Errorf("nearest document = %d, want %d (the paraphrase)", best, budgetDoc)
		}
	})

	t.Run("sparse finds the rare term", func(t *testing.T) {
		best := bestBy(t, len(docs.Data), func(i int) float64 {
			return dotProduct(queries.Data[1].Sparse, docs.Data[i].Sparse)
		})
		if best != quenselDoc {
			t.Errorf("highest sparse score = %d, want %d (the rare term)", best, quenselDoc)
		}
	})
}

func TestE2E_Representations_DeepInfraMultiVector(t *testing.T) {
	skipIfNoDeepInfraToken(t)
	ctx := representationCtx(t)

	// Multi-vector responses are large, so this runs against two documents
	// rather than the whole corpus.
	encoder, err := deepinfra.New(deepinfra.BGEM3Model)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	corpus := []string{representationDocs[0], representationDocs[budgetDoc]}
	docs, err := agentic.EncodeDocuments(ctx, encoder, corpus, agentic.RepresentationMultiVector)
	if err != nil {
		t.Fatalf("EncodeDocuments: %v", err)
	}
	queries, err := agentic.EncodeQueries(ctx, encoder, []string{paraphraseQuery}, agentic.RepresentationMultiVector)
	if err != nil {
		t.Fatalf("EncodeQueries: %v", err)
	}

	space, _ := docs.Space(agentic.RepresentationMultiVector)
	t.Logf("multi-vector space %s", space)

	// Token count varies with the document; token width must not.
	for i, item := range docs.Data {
		if len(item.MultiVector) == 0 {
			t.Fatalf("document %d has no token vectors", i)
		}
		for token, vec := range item.MultiVector {
			if len(vec) != space.Dimensions {
				t.Fatalf("document %d token %d width = %d, space declares %d",
					i, token, len(vec), space.Dimensions)
			}
		}
	}

	best := bestBy(t, len(docs.Data), func(i int) float64 {
		return maxSim(queries.Data[0].MultiVector, docs.Data[i].MultiVector)
	})
	if best != 1 {
		t.Errorf("MaxSim preferred document %d, want 1 (the paraphrase)", best)
	}
}

// The dense projection must behave exactly like any other Embedder, so an
// application can adopt this provider before it is ready to store sparse
// output.
func TestE2E_Representations_DeepInfraDenseEmbedderCompatibility(t *testing.T) {
	skipIfNoDeepInfraToken(t)
	ctx := representationCtx(t)

	encoder, err := deepinfra.New(deepinfra.BGEM3Model)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	checkEmbedder(t, ctx, encoder)
}
