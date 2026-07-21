package cohere

import (
	"net/http"
	"strings"
	"testing"
	"time"
)

// TestRetryDelayIsCapped pins the backoff schedule, including the cap that
// stops a long retry budget from turning into a multi-minute stall.
func TestRetryDelayIsCapped(t *testing.T) {
	tests := []struct {
		attempt int
		want    time.Duration
	}{
		{attempt: 0, want: 1 * time.Second},
		{attempt: 1, want: 2 * time.Second},
		{attempt: 2, want: 4 * time.Second},
		{attempt: 3, want: 4 * time.Second},
		{attempt: 8, want: 4 * time.Second},
	}

	for _, tt := range tests {
		if got := retryDelay(tt.attempt); got != tt.want {
			t.Errorf("retryDelay(%d) = %v, want %v", tt.attempt, got, tt.want)
		}
	}
}

// TestRetryableStatus pins which statuses are worth a second attempt. A 4xx
// other than 429 means the request itself is wrong, so resending it verbatim
// can only waste the caller's latency budget.
func TestRetryableStatus(t *testing.T) {
	retryable := []int{
		http.StatusTooManyRequests,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout,
	}
	for _, status := range retryable {
		if !retryableStatus(status) {
			t.Errorf("retryableStatus(%d) = false, want true", status)
		}
	}

	terminal := []int{
		http.StatusOK,
		http.StatusBadRequest,
		http.StatusUnauthorized,
		http.StatusForbidden,
		http.StatusNotFound,
		http.StatusUnprocessableEntity,
		http.StatusNotImplemented,
	}
	for _, status := range terminal {
		if retryableStatus(status) {
			t.Errorf("retryableStatus(%d) = true, want false", status)
		}
	}
}

// TestNewClientDefaults pins the values applied when a constructor is given
// nothing but a key.
func TestNewClientDefaults(t *testing.T) {
	c, err := newClient("k", "", nil, nil)
	if err != nil {
		t.Fatalf("newClient: %v", err)
	}
	if c.baseURL != defaultBaseURL {
		t.Errorf("baseURL = %q, want %q", c.baseURL, defaultBaseURL)
	}
	if c.maxRetries != defaultMaxRetries {
		t.Errorf("maxRetries = %d, want %d", c.maxRetries, defaultMaxRetries)
	}
	if c.httpClient == nil {
		t.Fatal("httpClient should be defaulted")
	}
	if c.httpClient.Timeout != defaultTimeout {
		t.Errorf("timeout = %v, want %v", c.httpClient.Timeout, defaultTimeout)
	}
}

// TestNewClientHonorsOverrides pins that supplied settings are not overwritten
// by the defaults, including a zero retry budget, which must survive rather
// than being treated as unset.
func TestNewClientHonorsOverrides(t *testing.T) {
	supplied := &http.Client{Timeout: time.Second}
	zero := 0

	c, err := newClient("k", "https://proxy.internal", supplied, &zero)
	if err != nil {
		t.Fatalf("newClient: %v", err)
	}
	if c.baseURL != "https://proxy.internal" {
		t.Errorf("baseURL = %q, want the supplied one", c.baseURL)
	}
	if c.httpClient != supplied {
		t.Error("httpClient should be the supplied client")
	}
	if c.maxRetries != 0 {
		t.Errorf("maxRetries = %d, want 0 (an explicit zero must not fall back to the default)", c.maxRetries)
	}
}

// TestStatusErrorPrefersMessage pins the error rendering for Cohere's
// {"id","message"} envelope and its fallbacks.
func TestStatusErrorPrefersMessage(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		want    string
	}{
		{name: "message", payload: `{"id":"r","message":"too many texts"}`, want: "too many texts"},
		{name: "empty message falls back to the body", payload: `{"id":"r","message":""}`, want: `{"id":"r","message":""}`},
		{name: "not JSON falls back to the body", payload: `service unavailable`, want: "service unavailable"},
		{name: "JSON of the wrong shape falls back", payload: `["a"]`, want: `["a"]`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := statusError("/v2/embed", 400, []byte(tt.payload))
			if err == nil {
				t.Fatal("statusError should always return an error")
			}
			got := err.Error()
			if !strings.Contains(got, tt.want) {
				t.Errorf("error = %q, want it to contain %q", got, tt.want)
			}
			if !strings.Contains(got, "/v2/embed") || !strings.Contains(got, "400") {
				t.Errorf("error = %q, want it to name the path and status", got)
			}
		})
	}
}

// TestInputTypeValid pins the accepted vocabulary directly, including the
// values Cohere supports but this package does not expose as constants.
func TestInputTypeValid(t *testing.T) {
	valid := []InputType{
		InputTypeSearchQuery,
		InputTypeSearchDocument,
		InputTypeClassification,
		InputTypeClustering,
	}
	for _, inputType := range valid {
		if !inputType.valid() {
			t.Errorf("%q.valid() = false, want true", inputType)
		}
	}

	invalid := []InputType{"", "image", "search-query", "SEARCH_QUERY", "document"}
	for _, inputType := range invalid {
		if inputType.valid() {
			t.Errorf("%q.valid() = true, want false", inputType)
		}
	}
}

// TestPostMarshalFailure pins that a body which cannot be marshaled is an error
// rather than a panic. No exported path can reach this today, but post is the
// shared entry point for both endpoints and a future field could.
func TestPostMarshalFailure(t *testing.T) {
	c, err := newClient("k", "", nil, nil)
	if err != nil {
		t.Fatalf("newClient: %v", err)
	}

	// A channel has no JSON representation.
	if _, err := c.post(t.Context(), "/v2/embed", make(chan int)); err == nil {
		t.Fatal("post should fail when the body cannot be marshaled")
	}
}
