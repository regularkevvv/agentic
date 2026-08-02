package onnx

import "sort"

// maxPadRatio bounds how far a row may be padded relative to its own length.
//
// A group's rows are consecutive in length order, so the group's width is its
// last row's length and every row is at least as long as its first. Closing the
// group once the width exceeds twice the shortest row means no row is ever
// padded past twice its own length, which caps the share of a forward pass
// spent on padding at one half.
//
// That bound is what keeps batching from being a pessimization. Measured on
// 2026-08-01, three inputs of 3, 3, and 18 tokens padded to a common width of
// 18 took 20 ms as one call and 13 ms as three: the two short rows each paid for
// 15 positions of nothing. Under this rule they form one group of width 3 and
// the long row runs alone, for 24 padded positions instead of 54.
const maxPadRatio = 2

// bytesPerLogit is the width of one float32 logit, named so the budget
// arithmetic below reads as the memory calculation it is.
const bytesPerLogit = 4

// groupByWidth orders inputs by token length and splits them into the forward
// passes that will encode them, returning input positions rather than lengths
// so the caller can scatter results back into request order.
//
// Sorting is not a heuristic here. For any partition into groups of a given
// size, ordering by length minimizes the total padded width, because a group's
// width is its longest member and neighbors in sorted order are the closest
// lengths available.
//
// Two rules close a group: the padding bound above, and budget, the ceiling in
// bytes on the logits tensor one pass may allocate — batch × width × vocabulary
// × 4. A single row that exceeds the budget alone is still emitted, because
// refusing to encode an input on account of a memory ceiling is not a useful
// failure and one row is the smallest pass there is.
func groupByWidth(lengths []int, vocabulary, budget int) [][]int {
	order := make([]int, len(lengths))
	for i := range order {
		order[i] = i
	}
	// Stable, so equal lengths stay in request order and the grouping — and
	// therefore the forward-pass count reported in usage — is a function of the
	// request alone.
	sort.SliceStable(order, func(a, b int) bool {
		return lengths[order[a]] < lengths[order[b]]
	})

	var (
		groups   [][]int
		current  []int
		shortest int
	)
	for _, i := range order {
		width := lengths[i]
		if len(current) > 0 && (width > shortest*maxPadRatio || exceeds(len(current)+1, width, vocabulary, budget)) {
			groups = append(groups, current)
			current = nil
		}
		if len(current) == 0 {
			shortest = width
		}
		current = append(current, i)
	}
	if len(current) > 0 {
		groups = append(groups, current)
	}
	return groups
}

// exceeds reports whether a pass of this shape would allocate more than budget.
// The product is taken in int64 because its three terms are each unbounded by
// anything in this package and their product is not obviously inside an int.
func exceeds(batch, width, vocabulary, budget int) bool {
	return int64(batch)*int64(width)*int64(vocabulary)*bytesPerLogit > int64(budget)
}
