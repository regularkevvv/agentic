package providerhttp

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
)

// newTestClient builds a client against handler with a backoff short enough
// that the retry paths run in test time rather than wall time.
func newTestClient(t *testing.T, handler http.HandlerFunc, cfg Config) (*Client, string) {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	if cfg.Token == "" {
		cfg.Token = "test-token"
	}
	client, err := New("testprovider", cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	client.Backoff = func(int, http.Header) time.Duration { return time.Millisecond }
	return client, server.URL
}

func TestNewValidatesConfiguration(t *testing.T) {
	negative := -1
	if _, err := New("p", Config{MaxRetries: &negative}); err == nil ||
		!strings.Contains(err.Error(), "p: max retries") {
		t.Errorf("got %v, want the provider-prefixed retry error", err)
	}

	zero := int64(0)
	if _, err := New("p", Config{MaxResponseBytes: &zero}); err == nil ||
		!strings.Contains(err.Error(), "p: max response bytes") {
		t.Errorf("got %v, want the provider-prefixed size error", err)
	}

	client, err := New("p", Config{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if client.maxRetries != 2 || client.maxResponseBytes != defaultMaxResponseBytes {
		t.Errorf("defaults = %d retries, %d bytes", client.maxRetries, client.maxResponseBytes)
	}
}

func TestPostSendsBearerTokenAndJSON(t *testing.T) {
	var authorization, contentType string
	client, url := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		contentType = r.Header.Get("Content-Type")
		fmt.Fprint(w, `{"ok":true}`)
	}, Config{Token: "secret"})

	payload, err := client.Post(context.Background(), url, map[string]string{"k": "v"})
	if err != nil {
		t.Fatalf("Post: %v", err)
	}
	if string(payload) != `{"ok":true}` {
		t.Errorf("payload = %q", payload)
	}
	if authorization != "Bearer secret" {
		t.Errorf("Authorization = %q", authorization)
	}
	if contentType != "application/json" {
		t.Errorf("Content-Type = %q", contentType)
	}
}

// An empty token is an anonymous request rather than an empty credential: a
// server that checks none must not be sent "Bearer ".
func TestPostWithoutTokenSendsNoAuthorizationHeader(t *testing.T) {
	var present bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, present = r.Header["Authorization"]
		fmt.Fprint(w, `{}`)
	}))
	t.Cleanup(server.Close)

	client, err := New("p", Config{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := client.Post(context.Background(), server.URL, map[string]string{}); err != nil {
		t.Fatalf("Post: %v", err)
	}
	if present {
		t.Error("an unauthenticated client sent an Authorization header")
	}
}

func TestPostRejectsUnmarshalableBody(t *testing.T) {
	client, url := newTestClient(t, func(http.ResponseWriter, *http.Request) {}, Config{})
	if _, err := client.Post(context.Background(), url, make(chan int)); err == nil {
		t.Fatal("a body that cannot be marshaled should fail before any request")
	}
}

func TestPostRejectsUnusableURLs(t *testing.T) {
	client, _ := newTestClient(t, func(http.ResponseWriter, *http.Request) {}, Config{})
	if _, err := client.Post(context.Background(), "http://e/%zz", map[string]string{}); err == nil {
		t.Fatal("a URL that cannot be parsed should fail rather than be dialed")
	}
}

func TestPostRejectsOversizedResponses(t *testing.T) {
	limit := int64(128)
	client, url := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, strings.Repeat("x", 4096))
	}, Config{MaxResponseBytes: &limit})

	_, err := client.Post(context.Background(), url, map[string]string{})
	if err == nil || !strings.Contains(err.Error(), "testprovider: response exceeds the 128 byte limit") {
		t.Fatalf("got %v, want the bounded-read error", err)
	}
}

func TestRetriesRateLimitThenSucceeds(t *testing.T) {
	var calls atomic.Int32
	client, url := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		fmt.Fprint(w, `{}`)
	}, Config{})

	if _, err := client.Post(context.Background(), url, map[string]string{}); err != nil {
		t.Fatalf("Post: %v", err)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("made %d calls, want a retry after the 429", got)
	}
}

// A cold endpoint answers 503 while the model loads, so that status has to be
// transient rather than fatal.
func TestRetriesColdEndpoint(t *testing.T) {
	var calls atomic.Int32
	client, url := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprint(w, `{"error": "Model is loading"}`)
			return
		}
		fmt.Fprint(w, `{}`)
	}, Config{})

	if _, err := client.Post(context.Background(), url, map[string]string{}); err != nil {
		t.Fatalf("Post: %v", err)
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("made %d calls, want two retries while the model loaded", got)
	}
}

func TestExhaustsRetryBudget(t *testing.T) {
	var calls atomic.Int32
	retries := 1
	client, url := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusBadGateway)
	}, Config{MaxRetries: &retries})

	_, err := client.Post(context.Background(), url, map[string]string{})
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Status != http.StatusBadGateway {
		t.Fatalf("got %v, want an APIError with status 502", err)
	}
	if apiErr.Provider != "testprovider" {
		t.Errorf("provider = %q, want the constructing package", apiErr.Provider)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("made %d calls, want one attempt plus one retry", got)
	}
}

func TestDoesNotRetryClientErrors(t *testing.T) {
	var calls atomic.Int32
	client, url := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error": "invalid token"}`)
	}, Config{})

	_, err := client.Post(context.Background(), url, map[string]string{})
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
	client, url := newTestClient(t, func(http.ResponseWriter, *http.Request) {}, Config{})
	client.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("connection reset")
	})}

	_, err := client.Post(context.Background(), url, map[string]string{})
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

	client, url := newTestClient(t, func(http.ResponseWriter, *http.Request) {}, Config{})
	client.httpClient = &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		calls.Add(1)
		cancel()
		return nil, r.Context().Err()
	})}

	_, err := client.Post(ctx, url, map[string]string{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("made %d attempts after cancellation, want 1", got)
	}
}

// Canceling from the backoff rather than from the handler is what puts the
// cancellation inside the wait. A context already canceled by the time the
// response is read fails in the transport instead, which is the previous test
// and a different branch.
func TestCancellationDuringBackoffStopsRetrying(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var calls, waits atomic.Int32

	client, url := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
	}, Config{})
	client.Backoff = func(_ int, header http.Header) time.Duration {
		waits.Add(1)
		if header == nil {
			t.Error("the status path should pass the response headers to the backoff")
		}
		cancel()
		return time.Minute
	}

	_, err := client.Post(ctx, url, map[string]string{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("made %d calls, want the backoff to abort after the first", got)
	}
	if got := waits.Load(); got != 1 {
		t.Errorf("entered the backoff %d times, want exactly one", got)
	}
}

func TestCancellationDuringTransportBackoffStopsRetrying(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var calls atomic.Int32

	client, url := newTestClient(t, func(http.ResponseWriter, *http.Request) {}, Config{})
	client.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("connection reset")
	})}
	client.Backoff = func(int, http.Header) time.Duration {
		cancel()
		return time.Minute
	}

	_, err := client.Post(ctx, url, map[string]string{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("made %d attempts, want the backoff to abort after the first", got)
	}
}

func TestResponseBodiesAreClosed(t *testing.T) {
	var opened, closed atomic.Int32

	client, url := newTestClient(t, func(http.ResponseWriter, *http.Request) {}, Config{})
	client.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		opened.Add(1)
		status := http.StatusTooManyRequests
		if opened.Load() > 1 {
			status = http.StatusOK
		}
		return &http.Response{
			StatusCode: status,
			Header:     make(http.Header),
			Body:       &trackedBody{Reader: strings.NewReader("{}"), onClose: func() { closed.Add(1) }},
		}, nil
	})}

	if _, err := client.Post(context.Background(), url, map[string]string{}); err != nil {
		t.Fatalf("Post: %v", err)
	}
	if opened.Load() != closed.Load() {
		t.Errorf("opened %d bodies but closed %d", opened.Load(), closed.Load())
	}
}

func TestReadAndCloseErrorsSurface(t *testing.T) {
	t.Run("read error", func(t *testing.T) {
		client, url := newTestClient(t, func(http.ResponseWriter, *http.Request) {}, Config{})
		client.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       &trackedBody{Reader: failingReader{}, onClose: func() {}},
			}, nil
		})}
		_, err := client.Post(context.Background(), url, map[string]string{})
		if err == nil || !strings.Contains(err.Error(), "read failed") {
			t.Fatalf("got %v, want the read error", err)
		}
	})

	t.Run("close error", func(t *testing.T) {
		client, url := newTestClient(t, func(http.ResponseWriter, *http.Request) {}, Config{})
		client.httpClient = &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body: &trackedBody{
					Reader:   strings.NewReader("{}"),
					closeErr: errors.New("close failed"),
					onClose:  func() {},
				},
			}, nil
		})}
		_, err := client.Post(context.Background(), url, map[string]string{})
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
			err := newAPIError("p", http.StatusBadGateway, "https://endpoint/x", []byte(tc.body))
			if !strings.Contains(err.Detail, tc.want) {
				t.Fatalf("detail = %q, want it to contain %q", err.Detail, tc.want)
			}
			if !strings.HasPrefix(err.Error(), "p: status 502") {
				t.Errorf("message = %q, want the provider and status first", err.Error())
			}
			if !strings.Contains(err.Error(), "https://endpoint/x") {
				t.Errorf("message = %q, want the endpoint named", err.Error())
			}
		})
	}

	t.Run("bounded detail", func(t *testing.T) {
		long := strings.Repeat("x", maxDetailLength*2)
		err := newAPIError("p", http.StatusBadRequest, "https://e", []byte(fmt.Sprintf(`{"error": %q}`, long)))
		if len(err.Detail) > maxDetailLength+3 || !strings.HasSuffix(err.Detail, "...") {
			t.Fatalf("detail is %d characters, want it bounded and marked", len(err.Detail))
		}
	})

	// A signed access parameter in the query string must not reach the error.
	t.Run("query string stripped", func(t *testing.T) {
		err := newAPIError("p", http.StatusForbidden, "https://e/path?token=secret", []byte(`{}`))
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
