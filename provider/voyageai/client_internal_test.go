package voyageai

import (
	"net/http"
	"testing"
	"time"
)

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 7, 20, 12, 0, 0, 0, time.UTC)

	tests := []struct {
		name  string
		value string
		want  time.Duration
		ok    bool
	}{
		{name: "absent", value: "", want: 0, ok: false},
		{name: "malformed", value: "soon", want: 0, ok: false},
		{name: "seconds", value: "7", want: 7 * time.Second, ok: true},
		{name: "zero seconds", value: "0", want: 0, ok: true},
		{name: "negative seconds clamps to zero", value: "-5", want: 0, ok: true},
		{name: "seconds above the cap", value: "600", want: maxRetryAfter, ok: true},
		{
			name:  "http date",
			value: "Mon, 20 Jul 2026 12:00:09 GMT",
			want:  9 * time.Second,
			ok:    true,
		},
		{
			name:  "http date in the past clamps to zero",
			value: "Mon, 20 Jul 2026 11:59:00 GMT",
			want:  0,
			ok:    true,
		},
		{
			name:  "http date above the cap",
			value: "Mon, 20 Jul 2026 12:10:00 GMT",
			want:  maxRetryAfter,
			ok:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := parseRetryAfter(tt.value, now)
			if ok != tt.ok {
				t.Fatalf("parseRetryAfter(%q) ok = %v, want %v", tt.value, ok, tt.ok)
			}
			if got != tt.want {
				t.Errorf("parseRetryAfter(%q) = %v, want %v", tt.value, got, tt.want)
			}
		})
	}
}

// TestWithJitterBounds pins that jitter only ever extends a delay, and never
// past 25% of it, so a retry can be reasoned about as a bounded wait.
func TestWithJitterBounds(t *testing.T) {
	for _, base := range []time.Duration{0, -time.Second, time.Millisecond, time.Second, 4 * time.Second} {
		if base <= 0 {
			if got := withJitter(base); got != 0 {
				t.Errorf("withJitter(%v) = %v, want 0", base, got)
			}
			continue
		}
		for range 200 {
			got := withJitter(base)
			if got < base || got > base+base/4+1 {
				t.Fatalf("withJitter(%v) = %v, out of bounds", base, got)
			}
		}
	}
}

// TestWithJitterVaries pins that the jitter is actually random: a fixed delay
// that always produced the same wait would resynchronize throttled clients,
// which is the whole reason the jitter exists.
func TestWithJitterVaries(t *testing.T) {
	const base = time.Second
	first := withJitter(base)
	for range 500 {
		if withJitter(base) != first {
			return
		}
	}
	t.Fatalf("withJitter(%v) returned %v every time, want a spread", base, first)
}

// TestRetryWaitPrefersRetryAfter pins that a server's Retry-After replaces the
// exponential backoff rather than adding to it, and that a missing or
// unparseable header falls back to the backoff.
func TestRetryWaitPrefersRetryAfter(t *testing.T) {
	tests := []struct {
		name    string
		attempt int
		header  http.Header
		min     time.Duration
		max     time.Duration
	}{
		{
			name:    "no header falls back to backoff",
			attempt: 2,
			header:  nil,
			min:     4 * time.Second,
			max:     5*time.Second + 1,
		},
		{
			name:    "empty header falls back to backoff",
			attempt: 0,
			header:  http.Header{},
			min:     time.Second,
			max:     time.Second + time.Second/4 + 1,
		},
		{
			name:    "unparseable header falls back to backoff",
			attempt: 0,
			header:  http.Header{"Retry-After": []string{"whenever"}},
			min:     time.Second,
			max:     time.Second + time.Second/4 + 1,
		},
		{
			name:    "header shortens a long backoff",
			attempt: 5,
			header:  http.Header{"Retry-After": []string{"0"}},
			min:     0,
			max:     0,
		},
		{
			name:    "header lengthens a short backoff",
			attempt: 0,
			header:  http.Header{"Retry-After": []string{"10"}},
			min:     10 * time.Second,
			max:     12*time.Second + 500*time.Millisecond + 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := retryWait(tt.attempt, tt.header)
			if got < tt.min || got > tt.max {
				t.Errorf("retryWait(%d, %v) = %v, want within [%v, %v]", tt.attempt, tt.header, got, tt.min, tt.max)
			}
		})
	}
}

// TestNewClientSharedByEmbedderAndReranker pins the point of the refactor:
// both constructors resolve transport through the same path, so auth, base URL
// and retry policy cannot drift apart between the two endpoints.
func TestNewClientSharedByEmbedderAndReranker(t *testing.T) {
	t.Setenv("VOYAGE_API_KEY", "env-key")

	embedder, err := New("voyage-3.5")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	reranker, err := NewReranker("rerank-2.5")
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}

	if embedder.apiKey != reranker.apiKey {
		t.Errorf("apiKey embedder = %q, reranker = %q", embedder.apiKey, reranker.apiKey)
	}
	if embedder.baseURL != reranker.baseURL {
		t.Errorf("baseURL embedder = %q, reranker = %q", embedder.baseURL, reranker.baseURL)
	}
	if embedder.maxRetries != reranker.maxRetries {
		t.Errorf("maxRetries embedder = %d, reranker = %d", embedder.maxRetries, reranker.maxRetries)
	}
}
