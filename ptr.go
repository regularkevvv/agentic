package agentic

// Int returns a pointer to v.
//
// Optional request fields are pointers so that "unset" is distinguishable from
// a meaningful zero — a MaxTokens of 0 is not the same as leaving the provider
// to choose. This makes those fields writable inline:
//
//	req := &agentic.ChatRequest{MaxTokens: agentic.Int(1024)}
func Int(v int) *int { return &v }

// Float64 returns a pointer to v. See [Int] for why these fields are pointers.
//
//	req := &agentic.ChatRequest{Temperature: agentic.Float64(0.2)}
func Float64(v float64) *float64 { return &v }

// Bool returns a pointer to v. See [Int] for why these fields are pointers.
//
//	req := &agentic.EmbeddingRequest{Input: texts, Truncate: agentic.Bool(false)}
func Bool(v bool) *bool { return &v }

// String returns a pointer to v. See [Int] for why these fields are pointers.
func String(v string) *string { return &v }
