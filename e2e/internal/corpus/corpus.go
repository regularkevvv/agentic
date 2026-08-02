// Package corpus holds the documents and the scoring arithmetic that the live
// provider tests and the runnable examples both use.
//
// They score with the same three functions on purpose. An example that ranked
// documents by arithmetic the tests never exercised would be demonstrating a
// retrieval story nothing verifies, and the copies drifted before this package
// existed.
//
// Nothing here belongs in the library: Agentic returns vectors and the metric
// to compare them under, and never a score, because only a retrieval system
// knows how it ranks.
package corpus

import (
	"math"

	agentic "github.com/regularkevvv/agentic"
)

// Documents has one document per retrieval mode under test, and is small
// enough to read and large enough to rank wrongly if a representation is
// broken.
var Documents = []string{
	"Sourdough bread is made by fermenting dough with naturally occurring lactobacilli.",
	"Engineering spends two thousand dollars a month running its infrastructure.",
	"The quensel actuator must be recalibrated before every third launch.",
	"Mount Everest is the highest mountain above sea level.",
}

// The two documents the queries below are supposed to find.
//
// BudgetDoc answers ParaphraseQuery without sharing any of its wording, so
// only a dense match reaches it. QuenselDoc carries a coined term that appears
// in no other document and in no vocabulary, so only a lexical match reaches
// it.
const (
	BudgetDoc  = 1
	QuenselDoc = 2
)

const (
	ParaphraseQuery = "how much does the engineering team cost each month?"
	RareTermQuery   = "quensel actuator recalibration"
)

// Cosine compares direction, which is the metric a dense space declares.
func Cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// DotProduct scores a sparse query against a sparse document by summing the
// weights of the coordinates they share, which is what a sparse index
// computes. Both vectors are sorted by construction, so one pass over each is
// enough.
func DotProduct(query, document *agentic.SparseVector) float64 {
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

// MaxSim is ColBERT's late-interaction score: for each query token take its
// best-matching document token, then sum. It is the reason multi-vector output
// must never be averaged into one vector.
func MaxSim(query, document [][]float32) float64 {
	var total float64
	for _, q := range query {
		best := math.Inf(-1)
		for _, d := range document {
			if score := Cosine(q, d); score > best {
				best = score
			}
		}
		if !math.IsInf(best, -1) {
			total += best
		}
	}
	return total
}
