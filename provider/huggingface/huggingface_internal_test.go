package huggingface

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/internal/core"
)

const testProtocolResponse = `{
	"version": "agentic.representations.v1",
	"model": "BAAI/bge-m3",
	"spaces": {"dense": {"id":"d","provider":"custom","model":"BAAI/bge-m3","kind":"dense","dimensions":2,"metric":"cosine"}},
	"data": [{"dense": [0.1, 0.2]}],
	"usage": {"input_tokens": 3, "request_count": 1}
}`

func denseRequest() *core.RepresentationRequest {
	return &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	}
}

// newRetryDedicated builds an encoder with a backoff short enough that the
// retry paths run in test time rather than wall time.
func newRetryDedicated(t *testing.T, handler http.HandlerFunc, opts ...DedicatedOption) *DedicatedEncoder {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	opts = append([]DedicatedOption{WithToken("hf_test"), WithModel("BAAI/bge-m3")}, opts...)
	encoder, err := NewDedicated(server.URL, opts...)
	if err != nil {
		t.Fatalf("NewDedicated: %v", err)
	}
	encoder.backoff = func(int, http.Header) time.Duration { return time.Millisecond }
	return encoder
}

func TestRetriesRateLimitThenSucceeds(t *testing.T) {
	var calls atomic.Int32
	encoder := newRetryDedicated(t, func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		fmt.Fprint(w, testProtocolResponse)
	})

	if _, err := encoder.Encode(context.Background(), denseRequest()); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("made %d calls, want a retry after the 429", got)
	}
}

// A cold dedicated endpoint answers 503 while the model loads, so that status
// has to be transient rather than fatal.
func TestRetriesColdEndpoint(t *testing.T) {
	var calls atomic.Int32
	encoder := newRetryDedicated(t, func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprint(w, `{"error": "Model is loading"}`)
			return
		}
		fmt.Fprint(w, testProtocolResponse)
	})

	if _, err := encoder.Encode(context.Background(), denseRequest()); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("made %d calls, want two retries while the model loaded", got)
	}
}

func TestExhaustsRetryBudget(t *testing.T) {
	var calls atomic.Int32
	encoder := newRetryDedicated(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusBadGateway)
	}, WithMaxRetries(1))

	_, err := encoder.Encode(context.Background(), denseRequest())
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Status != http.StatusBadGateway {
		t.Fatalf("got %v, want an APIError with status 502", err)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("made %d calls, want one attempt plus one retry", got)
	}
}

func TestDoesNotRetryClientErrors(t *testing.T) {
	var calls atomic.Int32
	encoder := newRetryDedicated(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error": "invalid token"}`)
	})

	_, err := encoder.Encode(context.Background(), denseRequest())
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Status != http.StatusUnauthorized {
		t.Fatalf("got %v, want an APIError with status 401", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("made %d calls; a 401 will fail identically on a retry", got)
	}
}

func TestRetriesTransportFailures(t *testing.T) {
	var calls atomic.Int32
	encoder := newRetryDedicated(t, func(http.ResponseWriter, *http.Request) {})
	encoder.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("connection reset")
	})}

	_, err := encoder.Encode(context.Background(), denseRequest())
	if err == nil || !strings.Contains(err.Error(), "connection reset") {
		t.Fatalf("got %v, want the transport error", err)
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("made %d attempts, want 3", got)
	}
}

func TestCancellationDuringTransportKeepsItsCause(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var calls atomic.Int32

	encoder := newRetryDedicated(t, func(http.ResponseWriter, *http.Request) {})
	encoder.httpClient = &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		calls.Add(1)
		cancel()
		return nil, r.Context().Err()
	})}

	_, err := encoder.Encode(ctx, denseRequest())
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("made %d attempts after cancellation, want 1", got)
	}
}

func TestCancellationDuringBackoffStopsRetrying(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var calls atomic.Int32

	encoder := newRetryDedicated(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		cancel()
		w.WriteHeader(http.StatusTooManyRequests)
	})
	encoder.backoff = func(int, http.Header) time.Duration { return time.Minute }

	_, err := encoder.Encode(ctx, denseRequest())
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("made %d calls, want the backoff to abort after the first", got)
	}
}

func TestResponseBodiesAreClosed(t *testing.T) {
	var opened, closed atomic.Int32

	encoder := newRetryDedicated(t, func(http.ResponseWriter, *http.Request) {})
	encoder.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		opened.Add(1)
		status, body := http.StatusTooManyRequests, "{}"
		if opened.Load() > 1 {
			status, body = http.StatusOK, testProtocolResponse
		}
		return &http.Response{
			StatusCode: status,
			Header:     make(http.Header),
			Body:       &trackedBody{Reader: strings.NewReader(body), onClose: func() { closed.Add(1) }},
		}, nil
	})}

	if _, err := encoder.Encode(context.Background(), denseRequest()); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if opened.Load() != closed.Load() {
		t.Errorf("opened %d bodies but closed %d", opened.Load(), closed.Load())
	}
}

func TestReadAndCloseErrorsSurface(t *testing.T) {
	t.Run("read error", func(t *testing.T) {
		encoder := newRetryDedicated(t, func(http.ResponseWriter, *http.Request) {})
		encoder.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       &trackedBody{Reader: failingReader{}, onClose: func() {}},
			}, nil
		})}
		_, err := encoder.Encode(context.Background(), denseRequest())
		if err == nil || !strings.Contains(err.Error(), "read failed") {
			t.Fatalf("got %v, want the read error", err)
		}
	})

	t.Run("close error", func(t *testing.T) {
		encoder := newRetryDedicated(t, func(http.ResponseWriter, *http.Request) {})
		encoder.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body: &trackedBody{
					Reader:   strings.NewReader(testProtocolResponse),
					closeErr: errors.New("close failed"),
					onClose:  func() {},
				},
			}, nil
		})}
		_, err := encoder.Encode(context.Background(), denseRequest())
		if err == nil || !strings.Contains(err.Error(), "close failed") {
			t.Fatalf("got %v, want the close error", err)
		}
	})
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 8, 1, 12, 0, 0, 0, time.UTC)

	tests := []struct {
		name  string
		value string
		want  time.Duration
		ok    bool
	}{
		{"absent", "", 0, false},
		{"seconds", "5", 5 * time.Second, true},
		{"clamped seconds", "600", maxRetryAfter, true},
		{"http date", now.Add(3 * time.Second).Format(http.TimeFormat), 3 * time.Second, true},
		{"past http date", now.Add(-time.Minute).Format(http.TimeFormat), 0, true},
		{"clamped http date", now.Add(time.Hour).Format(http.TimeFormat), maxRetryAfter, true},
		{"garbage", "soon", 0, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := parseRetryAfter(tc.value, now)
			if ok != tc.ok || got != tc.want {
				t.Fatalf("parseRetryAfter(%q) = %v, %v; want %v, %v", tc.value, got, ok, tc.want, tc.ok)
			}
		})
	}
}

func TestRetryWaitUsesHeaderThenBackoff(t *testing.T) {
	if got := retryWait(0, http.Header{"Retry-After": []string{"2"}}); got < 2*time.Second {
		t.Errorf("retryWait with a Retry-After = %v, want at least 2s", got)
	}
	if got := retryWait(0, http.Header{}); got < time.Second {
		t.Errorf("retryWait without a header = %v, want the exponential floor", got)
	}
	if got := retryWait(9, nil); got < 4*time.Second || got > 6*time.Second {
		t.Errorf("retryWait(9) = %v, want the capped delay", got)
	}
	if withJitter(0) != 0 {
		t.Error("a non-positive delay should stay zero")
	}
	if got := retryDelay(1); got != 2*time.Second {
		t.Errorf("retryDelay(1) = %v", got)
	}
}

func TestSleepRetryRespectsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := sleepRetry(ctx, time.Hour); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if err := sleepRetry(context.Background(), time.Millisecond); err != nil {
		t.Fatalf("got %v, want a completed sleep", err)
	}
}

func TestNewAPIError(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{"error string", `{"error": "model not found"}`, "model not found"},
		{"error object", `{"error": {"code": 42}}`, `{"code":`},
		{"detail string", `{"detail": "bad request"}`, "bad request"},
		{"unparsable", `<html>bad gateway</html>`, ""},
		{"empty envelope", `{}`, ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := newAPIError(http.StatusBadGateway, "https://endpoint/x", []byte(tc.body))
			if !strings.Contains(err.Detail, tc.want) {
				t.Fatalf("detail = %q, want it to contain %q", err.Detail, tc.want)
			}
			if !strings.Contains(err.Error(), "https://endpoint/x") {
				t.Errorf("message = %q, want the endpoint named", err.Error())
			}
		})
	}

	t.Run("bounded detail", func(t *testing.T) {
		long := strings.Repeat("x", maxDetailLength*2)
		err := newAPIError(http.StatusBadRequest, "https://e", []byte(fmt.Sprintf(`{"error": %q}`, long)))
		if len(err.Detail) > maxDetailLength+3 || !strings.HasSuffix(err.Detail, "...") {
			t.Fatalf("detail is %d characters, want it bounded and marked", len(err.Detail))
		}
	})

	// A signed access parameter in the query string must not reach the error.
	t.Run("query string stripped", func(t *testing.T) {
		err := newAPIError(http.StatusForbidden, "https://e/path?token=secret", []byte(`{}`))
		if strings.Contains(err.Error(), "secret") {
			t.Fatalf("error leaks the query string: %q", err.Error())
		}
		if err.Endpoint != "https://e/path" {
			t.Errorf("endpoint = %q", err.Endpoint)
		}
	})
}

func TestStripQueryAndTruncate(t *testing.T) {
	if got := stripQuery("https://e/path"); got != "https://e/path" {
		t.Errorf("stripQuery kept %q", got)
	}
	if got := truncate("short", 10); got != "short" {
		t.Errorf("truncate kept %q", got)
	}
}

func TestInjectedHTTPClientsAreUsed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, testProtocolResponse)
	}))
	t.Cleanup(server.Close)

	t.Run("dedicated", func(t *testing.T) {
		var used atomic.Bool
		client := &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
			used.Store(true)
			return http.DefaultTransport.RoundTrip(r)
		})}
		encoder, err := NewDedicated(server.URL, WithToken("t"), WithHTTPClient(client))
		if err != nil {
			t.Fatalf("NewDedicated: %v", err)
		}
		if _, err := encoder.Encode(context.Background(), denseRequest()); err != nil {
			t.Fatalf("Encode: %v", err)
		}
		if !used.Load() {
			t.Error("the injected client was not used")
		}
	})

	vectorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `[[0.1, 0.2]]`)
	}))
	t.Cleanup(vectorServer.Close)

	t.Run("shared", func(t *testing.T) {
		var used atomic.Bool
		client := &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
			used.Store(true)
			return http.DefaultTransport.RoundTrip(r)
		})}
		encoder, err := NewShared("m", WithSharedToken("t"),
			WithRouterURL(vectorServer.URL), WithSharedHTTPClient(client))
		if err != nil {
			t.Fatalf("NewShared: %v", err)
		}
		if _, err := encoder.Encode(context.Background(), denseRequest()); err != nil {
			t.Fatalf("Encode: %v", err)
		}
		if !used.Load() {
			t.Error("the injected client was not used")
		}
	})
}

func TestConfiguredLimitsAreEnforced(t *testing.T) {
	twoInputs := &core.RepresentationRequest{
		Input:   []string{"a", "b"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	}

	t.Run("dedicated", func(t *testing.T) {
		encoder, err := NewDedicated("https://e", WithToken("t"),
			WithLimits(core.RepresentationLimits{MaxInputs: 1}))
		if err != nil {
			t.Fatalf("NewDedicated: %v", err)
		}
		if _, err := encoder.Encode(context.Background(), twoInputs); !errors.Is(err, core.ErrInvalidRepresentationRequest) {
			t.Fatalf("got %v, want the configured limit to be enforced", err)
		}
	})

	t.Run("shared", func(t *testing.T) {
		encoder, err := NewShared("m", WithSharedToken("t"),
			WithSharedLimits(core.RepresentationLimits{MaxInputs: 1}))
		if err != nil {
			t.Fatalf("NewShared: %v", err)
		}
		if _, err := encoder.Encode(context.Background(), twoInputs); !errors.Is(err, core.ErrInvalidRepresentationRequest) {
			t.Fatalf("got %v, want the configured limit to be enforced", err)
		}
	})
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type trackedBody struct {
	io.Reader
	closeErr error
	onClose  func()
}

func (b *trackedBody) Close() error {
	b.onClose()
	return b.closeErr
}

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, errors.New("read failed") }
