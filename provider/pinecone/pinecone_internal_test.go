package pinecone

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

const testDenseResponse = `{"model":"m","vector_type":"dense","data":[{"vector_type":"dense","values":[0.1,0.2]}],"usage":{"total_tokens":3}}`

func denseRequest() *core.RepresentationRequest {
	return &core.RepresentationRequest{
		Input:   []string{"a"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	}
}

// newRetryEncoder builds an encoder with a backoff short enough that the retry
// paths run in test time rather than wall time.
func newRetryEncoder(t *testing.T, handler http.HandlerFunc, opts ...Option) *Encoder {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	opts = append([]Option{WithAPIKey("pc-test"), WithBaseURL(server.URL)}, opts...)
	encoder, err := New("llama-text-embed-v2", opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	encoder.backoff = func(int, http.Header) time.Duration { return time.Millisecond }
	return encoder
}

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

	if _, err := encoder.Encode(context.Background(), denseRequest()); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("made %d calls, want a retry after the 429", got)
	}
}

func TestRetriesTransientServerErrors(t *testing.T) {
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
			}, WithMaxRetries(1))

			_, err := encoder.Encode(context.Background(), denseRequest())
			var apiErr *APIError
			if !errors.As(err, &apiErr) || apiErr.Status != status {
				t.Fatalf("got %v, want an APIError with status %d", err, status)
			}
			if got := calls.Load(); got != 2 {
				t.Errorf("made %d calls, want one attempt plus one retry", got)
			}
		})
	}
}

func TestDoesNotRetryClientErrors(t *testing.T) {
	var calls atomic.Int32
	encoder := newRetryEncoder(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error":{"code":"UNAUTHENTICATED","message":"invalid key"}}`)
	})

	_, err := encoder.Encode(context.Background(), denseRequest())
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Code != "UNAUTHENTICATED" {
		t.Fatalf("got %v, want an UNAUTHENTICATED APIError", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("made %d calls; a 401 will fail identically on a retry", got)
	}
}

func TestRetriesTransportFailures(t *testing.T) {
	var calls atomic.Int32
	encoder := newRetryEncoder(t, func(http.ResponseWriter, *http.Request) {})
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

func TestTransportFailureIsReturnedWhenRetriesAreOff(t *testing.T) {
	var calls atomic.Int32
	encoder := newRetryEncoder(t, func(http.ResponseWriter, *http.Request) {}, WithMaxRetries(0))
	encoder.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("connection refused")
	})}

	if _, err := encoder.Encode(context.Background(), denseRequest()); err == nil {
		t.Fatal("expected the transport error")
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("made %d attempts with retries disabled, want 1", got)
	}
}

func TestCancellationDuringTransportKeepsItsCause(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var calls atomic.Int32

	encoder := newRetryEncoder(t, func(http.ResponseWriter, *http.Request) {})
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

	encoder := newRetryEncoder(t, func(w http.ResponseWriter, _ *http.Request) {
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

	encoder := newRetryEncoder(t, func(http.ResponseWriter, *http.Request) {})
	encoder.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		opened.Add(1)
		status, body := http.StatusTooManyRequests, "{}"
		if opened.Load() > 1 {
			status, body = http.StatusOK, testDenseResponse
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
		encoder := newRetryEncoder(t, func(http.ResponseWriter, *http.Request) {})
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
		t.Errorf("retryWait without a header = %v", got)
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
		code string
		want string
	}{
		{"structured", `{"error":{"code":"INVALID_ARGUMENT","message":"bad input"}}`, "INVALID_ARGUMENT", "bad input"},
		{"bare message", `{"message":"overloaded"}`, "", "overloaded"},
		{"unparsable", `<html>bad gateway</html>`, "", ""},
		{"empty envelope", `{}`, "", ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := newAPIError(http.StatusBadGateway, []byte(tc.body))
			if err.Code != tc.code || err.Detail != tc.want {
				t.Fatalf("error = %+v, want code %q and detail %q", err, tc.code, tc.want)
			}
			if !strings.Contains(err.Error(), "status 502") {
				t.Errorf("message = %q", err.Error())
			}
		})
	}

	t.Run("bounded detail", func(t *testing.T) {
		long := strings.Repeat("x", maxDetailLength*2)
		err := newAPIError(http.StatusBadRequest, []byte(fmt.Sprintf(`{"error":{"message":%q}}`, long)))
		if len(err.Detail) > maxDetailLength+3 || !strings.HasSuffix(err.Detail, "...") {
			t.Fatalf("detail is %d characters, want it bounded and marked", len(err.Detail))
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

func TestOptionsReachTheRequest(t *testing.T) {
	var apiVersion string
	var used atomic.Bool

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		apiVersion = r.Header.Get("X-Pinecone-Api-Version")
		fmt.Fprint(w, testDenseResponse)
	}))
	t.Cleanup(server.Close)

	client := &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		used.Store(true)
		return http.DefaultTransport.RoundTrip(r)
	})}

	encoder, err := New("llama-text-embed-v2",
		WithAPIKey("k"),
		WithBaseURL(server.URL),
		WithAPIVersion("2099-01"),
		WithHTTPClient(client),
		WithModelRevision("rev-1", "tok-1"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := encoder.Encode(context.Background(), denseRequest())
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	if apiVersion != "2099-01" {
		t.Errorf("API version header = %q", apiVersion)
	}
	if !used.Load() {
		t.Error("the injected HTTP client was not used")
	}
	space := resp.Spaces[core.RepresentationDense]
	if space.Revision != "rev-1" || space.Tokenizer != "tok-1" {
		t.Errorf("space did not record the configured revisions: %+v", space)
	}
}

// A recorded revision is what lets a consumer notice that the model behind a
// name changed.
func TestModelRevisionChangesSpaceIdentity(t *testing.T) {
	spaceID := func(opts ...Option) string {
		encoder := newRetryEncoder(t, func(w http.ResponseWriter, _ *http.Request) {
			fmt.Fprint(w, testDenseResponse)
		}, opts...)
		resp, err := encoder.Encode(context.Background(), denseRequest())
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		return resp.Spaces[core.RepresentationDense].ID
	}
	if spaceID() == spaceID(WithModelRevision("rev-1", "tok-1")) {
		t.Fatal("recording a revision did not change the space ID")
	}
}

func TestConfiguredLimitsAreEnforced(t *testing.T) {
	encoder, err := New("m", WithAPIKey("k"), WithLimits(core.RepresentationLimits{MaxInputs: 1}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = encoder.Encode(context.Background(), &core.RepresentationRequest{
		Input:   []string{"a", "b"},
		Outputs: []core.RepresentationKind{core.RepresentationDense},
	})
	if !errors.Is(err, core.ErrInvalidRepresentationRequest) {
		t.Fatalf("got %v, want the configured limit to be enforced", err)
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
