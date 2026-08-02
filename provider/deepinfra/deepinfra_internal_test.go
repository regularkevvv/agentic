package deepinfra

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

	"github.com/regularkevvv/agentic/internal/retrieval"
)

const testDenseResponse = `{"embeddings": [[0.1, 0.2]], "input_tokens": 3}`

// newRetryEncoder builds an encoder against handler with a backoff short
// enough that the retry paths run in test time rather than in wall time.
func newRetryEncoder(t *testing.T, handler http.HandlerFunc, opts ...Option) *Encoder {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	opts = append([]Option{WithAPIToken("test-token"), WithBaseURL(server.URL)}, opts...)
	encoder, err := New(BGEM3Model, opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	encoder.backoff = func(int, http.Header) time.Duration { return time.Millisecond }
	return encoder
}

func denseOnly() *retrieval.RepresentationRequest {
	return &retrieval.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []retrieval.RepresentationKind{RepresentationDenseForTest},
	}
}

// RepresentationDenseForTest keeps the request builder above readable without
// importing the core constant at every call site.
const RepresentationDenseForTest = retrieval.RepresentationDense

func TestRetriesRateLimitThenSucceeds(t *testing.T) {
	var calls atomic.Int32
	encoder := newRetryEncoder(t, func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		fmt.Fprint(w, testDenseResponse)
	})

	if _, err := encoder.Encode(context.Background(), denseOnly()); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("made %d calls, want a retry after the 429", got)
	}
}

func TestRetriesTransientServerErrorsWithinBudget(t *testing.T) {
	for _, status := range []int{
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout,
	} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			var calls atomic.Int32
			encoder := newRetryEncoder(t, func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				w.WriteHeader(status)
			}, WithMaxRetries(2))

			_, err := encoder.Encode(context.Background(), denseOnly())
			var apiErr *APIError
			if !errors.As(err, &apiErr) || apiErr.Status != status {
				t.Fatalf("got %v, want an APIError with status %d", err, status)
			}
			if got := calls.Load(); got != 3 {
				t.Errorf("made %d calls, want 3 (one attempt plus two retries)", got)
			}
		})
	}
}

func TestRetriesTransportFailures(t *testing.T) {
	var calls atomic.Int32
	encoder := newRetryEncoder(t, func(http.ResponseWriter, *http.Request) {})
	encoder.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("connection reset")
	})}

	_, err := encoder.Encode(context.Background(), denseOnly())
	if err == nil || !strings.Contains(err.Error(), "connection reset") {
		t.Fatalf("got %v, want the transport error", err)
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("made %d attempts, want 3", got)
	}
}

func TestTransportFailureIsReturnedWhenRetriesAreOff(t *testing.T) {
	var calls atomic.Int32
	encoder := newRetryEncoder(t, func(http.ResponseWriter, *http.Request) {}, WithMaxRetries(0))
	encoder.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("connection refused")
	})}

	_, err := encoder.Encode(context.Background(), denseOnly())
	if err == nil || !strings.Contains(err.Error(), "connection refused") {
		t.Fatalf("got %v, want the transport error", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("made %d attempts with retries disabled, want 1", got)
	}
}

func TestWithHTTPClientIsUsed(t *testing.T) {
	var used atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, testDenseResponse)
	}))
	t.Cleanup(server.Close)

	custom := &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		used.Store(true)
		return http.DefaultTransport.RoundTrip(r)
	})}
	encoder, err := New(BGEM3Model, WithAPIToken("t"), WithBaseURL(server.URL), WithHTTPClient(custom))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := encoder.Encode(context.Background(), denseOnly()); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if !used.Load() {
		t.Error("the injected HTTP client was not used")
	}
}

// A canceled context must surface as context.Canceled, not as a generic
// transport failure that a retry wrapper above would treat as transient.
func TestCancellationDuringTransportKeepsItsCause(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var calls atomic.Int32

	encoder := newRetryEncoder(t, func(http.ResponseWriter, *http.Request) {})
	encoder.httpClient = &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		calls.Add(1)
		cancel()
		return nil, r.Context().Err()
	})}

	_, err := encoder.Encode(ctx, denseOnly())
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

	encoder := newRetryEncoder(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		cancel()
		w.WriteHeader(http.StatusTooManyRequests)
	})
	encoder.backoff = func(int, http.Header) time.Duration { return time.Minute }

	_, err := encoder.Encode(ctx, denseOnly())
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("made %d calls, want the backoff to abort after the first", got)
	}
}

// A response body left open leaks a connection from the pool on every retry.
func TestResponseBodiesAreClosed(t *testing.T) {
	var opened, closed atomic.Int32

	encoder := newRetryEncoder(t, func(http.ResponseWriter, *http.Request) {})
	encoder.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		opened.Add(1)
		status := http.StatusTooManyRequests
		body := "{}"
		if opened.Load() > 1 {
			status = http.StatusOK
			body = testDenseResponse
		}
		return &http.Response{
			StatusCode: status,
			Header:     make(http.Header),
			Body: &trackedBody{
				Reader: strings.NewReader(body),
				onClose: func() {
					closed.Add(1)
				},
			},
		}, nil
	})}

	if _, err := encoder.Encode(context.Background(), denseOnly()); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if opened.Load() != closed.Load() {
		t.Errorf("opened %d bodies but closed %d", opened.Load(), closed.Load())
	}
}

func TestReadAndCloseErrorsSurface(t *testing.T) {
	t.Run("read error", func(t *testing.T) {
		encoder := newRetryEncoder(t, func(http.ResponseWriter, *http.Request) {})
		encoder.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       &trackedBody{Reader: failingReader{}, onClose: func() {}},
			}, nil
		})}
		_, err := encoder.Encode(context.Background(), denseOnly())
		if err == nil || !strings.Contains(err.Error(), "read failed") {
			t.Fatalf("got %v, want the read error", err)
		}
	})

	t.Run("close error", func(t *testing.T) {
		encoder := newRetryEncoder(t, func(http.ResponseWriter, *http.Request) {})
		encoder.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body: &trackedBody{
					Reader:   strings.NewReader(testDenseResponse),
					closeErr: errors.New("close failed"),
					onClose:  func() {},
				},
			}, nil
		})}
		_, err := encoder.Encode(context.Background(), denseOnly())
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
	header := http.Header{"Retry-After": []string{"2"}}
	if got := retryWait(0, header); got < 2*time.Second {
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
		name      string
		body      string
		wantMsg   string
		wantEmpty bool
	}{
		{"string detail", `{"detail": "bad inputs"}`, "bad inputs", false},
		{"structured detail", `{"detail": [{"loc": ["body"]}]}`, `[{"loc":`, false},
		{"error field", `{"error": "overloaded"}`, "overloaded", false},
		{"unparsable body", `<html>gateway timeout</html>`, "", true},
		{"empty envelope", `{}`, "", true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := newAPIError(http.StatusBadGateway, []byte(tc.body))
			if tc.wantEmpty {
				if err.Detail != "" {
					t.Fatalf("detail = %q, want empty", err.Detail)
				}
				if !strings.Contains(err.Error(), "status 502") {
					t.Errorf("message = %q", err.Error())
				}
				return
			}
			if !strings.Contains(err.Detail, tc.wantMsg) {
				t.Fatalf("detail = %q, want it to contain %q", err.Detail, tc.wantMsg)
			}
		})
	}

	t.Run("request id", func(t *testing.T) {
		err := newAPIError(http.StatusInternalServerError, []byte(`{"request_id": "req_123", "detail": "boom"}`))
		if err.RequestID != "req_123" || !strings.Contains(err.Error(), "req_123") {
			t.Fatalf("request id = %q, message = %q", err.RequestID, err.Error())
		}
	})

	t.Run("bounded detail", func(t *testing.T) {
		long := strings.Repeat("x", maxDetailLength*2)
		err := newAPIError(http.StatusBadRequest, []byte(fmt.Sprintf(`{"detail": %q}`, long)))
		if len(err.Detail) > maxDetailLength+3 {
			t.Fatalf("detail is %d characters, want it bounded", len(err.Detail))
		}
		if !strings.HasSuffix(err.Detail, "...") {
			t.Error("a truncated detail should say so")
		}
	})
}

func TestTruncate(t *testing.T) {
	if got := truncate("short", 10); got != "short" {
		t.Errorf("truncate kept %q", got)
	}
	if got := truncate("abcdef", 3); got != "abc..." {
		t.Errorf("truncate = %q", got)
	}
}

func TestBaseURLTrailingSlashIsNormalized(t *testing.T) {
	var path string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		fmt.Fprint(w, testDenseResponse)
	}))
	t.Cleanup(server.Close)

	encoder, err := New(BGEM3Model, WithAPIToken("t"), WithBaseURL(server.URL+"/"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := encoder.Encode(context.Background(), denseOnly()); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if path != "/inference/"+BGEM3Model {
		t.Errorf("path = %q, want no doubled slash", path)
	}
}

func TestWithLimitsIsApplied(t *testing.T) {
	encoder, err := New(BGEM3Model,
		WithAPIToken("t"),
		WithLimits(retrieval.RepresentationLimits{MaxInputs: 1}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = encoder.Encode(context.Background(), &retrieval.RepresentationRequest{
		Input:   []string{"a", "b"},
		Outputs: []retrieval.RepresentationKind{retrieval.RepresentationDense},
	})
	if !errors.Is(err, retrieval.ErrInvalidRepresentationRequest) {
		t.Fatalf("got %v, want the configured limit to be enforced", err)
	}
}

func TestWidthHelpersHandleEmptyBatches(t *testing.T) {
	if width(nil) != 0 {
		t.Error("width of an empty batch should be 0")
	}
	if tokenWidth(nil) != 0 {
		t.Error("token width of an empty batch should be 0")
	}
	if tokenWidth([][][]float32{{}, {{1, 2}}}) != 2 {
		t.Error("token width should skip empty items to find a width")
	}
}

func TestDecodeSparseRejectsEmptyRow(t *testing.T) {
	var scratch []float32
	if _, _, err := decodeSparse([]byte("  "), 0, &scratch); err == nil {
		t.Fatal("an empty sparse row should be rejected")
	}
	if _, err := decodeSparseMap(nil, 0); err == nil {
		t.Fatal("an empty coordinate map should be rejected")
	}
	if _, err := decodeSparseMap([]byte(`[1]`), 0); err == nil {
		t.Fatal("an array is not a coordinate map")
	}
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
