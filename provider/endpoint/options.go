package endpoint

import (
	"net/http"

	"github.com/regularkevvv/agentic/internal/providerhttp"
	"github.com/regularkevvv/agentic/internal/retrieval"
)

// Option configures an Encoder.
type Option func(*config)

type config struct {
	providerhttp.Config

	// tokenSet records that WithToken was called, which an empty Token cannot:
	// a caller who passed one and got an empty string must not silently
	// inherit the environment's credential instead.
	tokenSet bool

	unauthenticated bool
	model           string
	batchSize       int
	outputs         []retrieval.RepresentationKind
	spaces          map[retrieval.RepresentationKind]retrieval.VectorSpace
	limits          *retrieval.RepresentationLimits
}

// WithToken sets the bearer token sent to the endpoint. If not set,
// AGENTIC_ENDPOINT_TOKEN is used.
//
// An empty token is an error, not an anonymous request and not a fallback to
// the environment: use [WithoutAuthentication] for an endpoint that checks no
// credential.
func WithToken(token string) Option {
	return func(c *config) {
		c.Token = token
		c.tokenSet = true
	}
}

// WithoutAuthentication sends no Authorization header.
//
// A handler on a loopback address, or behind a network boundary that already
// authorizes the caller, has no token to check. The alternative — passing an
// invented token — writes a credential that does not exist into whatever
// configuration file carries it, and hides the fact that nothing is being
// authenticated.
func WithoutAuthentication() Option {
	return func(c *config) { c.unauthenticated = true }
}

// WithHTTPClient sets a custom HTTP client, for proxies, instrumentation, or
// tests.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) { c.HTTPClient = client }
}

// WithMaxRetries sets how many times a request is retried on 429 and transient
// 5xx responses (default 2). A cold endpoint answers 503 while it loads, which
// counts as transient.
func WithMaxRetries(retries int) Option {
	return func(c *config) { c.MaxRetries = &retries }
}

// WithMaxResponseBytes caps the response body this encoder will read (default
// 64 MiB).
func WithMaxResponseBytes(limit int64) Option {
	return func(c *config) { c.MaxResponseBytes = &limit }
}

// WithModel records the model name the endpoint serves, used when the handler
// reports none.
func WithModel(model string) Option {
	return func(c *config) { c.model = model }
}

// WithBatchSize splits requests larger than size into that many inputs per
// call, preserving order and summing usage. Zero, the default, sends the batch
// as one request.
func WithBatchSize(size int) Option {
	return func(c *config) { c.batchSize = size }
}

// WithOutputs declares which representation kinds this endpoint serves
// (default all three, matching the reference handler).
func WithOutputs(kinds ...retrieval.RepresentationKind) Option {
	return func(c *config) { c.outputs = append([]retrieval.RepresentationKind(nil), kinds...) }
}

// WithVectorSpaces pins the vector spaces this endpoint encodes into.
//
// A pinned space must match what the handler reports, exactly; a kind the
// handler leaves undescribed is filled from here. Pin them for any endpoint
// whose output you intend to keep, so that a redeployment onto different
// weights fails loudly instead of quietly mixing two generations of vectors in
// one index.
func WithVectorSpaces(spaces map[retrieval.RepresentationKind]retrieval.VectorSpace) Option {
	return func(c *config) {
		c.spaces = make(map[retrieval.RepresentationKind]retrieval.VectorSpace, len(spaces))
		for kind, space := range spaces {
			c.spaces[kind] = space
		}
	}
}

// WithLimits overrides the request and response size ceilings.
func WithLimits(limits retrieval.RepresentationLimits) Option {
	return func(c *config) { c.limits = &limits }
}
