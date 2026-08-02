package onnx

import (
	"math"

	agentic "github.com/regularkevvv/agentic"
)

// pooler reduces a forward pass's logits to one sparse vector per row.
//
// The reduction is SPLADE's, and is the same four lines the model's own
// sentence-transformers pooling config declares: log1p(relu(x))
// so weights are non-negative and saturating, zero at padded positions, then
// the maximum over sequence positions for each vocabulary entry. A term scores
// highly if *any* position predicts it, which is how a word the text never
// contained acquires weight.
//
// Leaving this out of the exported graph is deliberate. It keeps what the model
// contributes separable from what this package contributes, so a disagreement
// with the PyTorch reference says which half moved.
type pooler struct {
	// weights is one vocabulary-wide accumulator, reused across every row of
	// every pass.
	//
	// Allocating it per row would cost 200 KB at Granite's vocabulary to keep
	// the few hundred coordinates that survive, and a batch of a hundred short
	// inputs would churn 20 MB for nothing. The same reasoning produced the
	// reused decode buffer in provider/deepinfra.
	weights []float32
}

// reduce pools one row's logits into a sparse vector.
//
// logits is that row's [sequence, vocabulary] slab, flattened; mask is its
// attention mask, one entry per position. The returned vector's indices are
// strictly increasing because the accumulator is walked in order, which is the
// canonical form the representation contract requires rather than a property to
// re-establish by sorting.
func (p *pooler) reduce(logits []float32, mask []int64, vocabulary int) *agentic.SparseVector {
	if cap(p.weights) < vocabulary {
		p.weights = make([]float32, vocabulary)
	}
	weights := p.weights[:vocabulary]
	clear(weights)

	for position, attended := range mask {
		if attended == 0 {
			continue
		}
		row := logits[position*vocabulary : (position+1)*vocabulary]
		for v, logit := range row {
			if logit <= 0 {
				continue // relu, and log1p of it would be zero anyway
			}
			// log1p in float64 and then rounded is correctly rounded, where a
			// float32 log1p is not; the difference is below the 4.6e-06 the
			// graph itself contributes, but it costs nothing to be on the
			// right side of it.
			if w := float32(math.Log1p(float64(logit))); w > weights[v] {
				weights[v] = w
			}
		}
	}

	vector := &agentic.SparseVector{}
	for v, weight := range weights {
		if weight > 0 {
			vector.Indices = append(vector.Indices, uint32(v))
			vector.Values = append(vector.Values, weight)
		}
	}
	return vector
}
