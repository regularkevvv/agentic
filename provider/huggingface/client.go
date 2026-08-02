// Package huggingface implements core.RepresentationEncoder over the Hugging
// Face Inference Providers router.
//
// [NewShared] calls the router's feature-extraction task, which returns one
// dense vector per input and nothing else. It is advertised as dense-only for
// exactly that reason. A model card saying the model produces sparse weights
// locally is not evidence that the hosted route returns them, and this package
// does not claim a capability it has not observed in a response.
//
// An Inference Endpoint you operate is not a Hugging Face API — it is your own
// handler on a URL Hugging Face happens to host — so it is served by
// [github.com/regularkevvv/agentic/provider/endpoint], which speaks the
// versioned protocol to any host that runs it.
package huggingface

import (
	"errors"
	"os"

	"github.com/regularkevvv/agentic/internal/providerhttp"
)

// defaultRouterURL is the Inference Providers router, which serves the shared
// hosted models.
const defaultRouterURL = "https://router.huggingface.co"

// providerName identifies this package in vector spaces and errors.
const providerName = "huggingface"

// APIError reports a non-200 response from Hugging Face.
//
// It is the shared transport's error type. It carries the status and Hugging
// Face's own bounded error message, never the raw body: a rejected request's
// error quotes the offending input back at you, and an error that may contain
// a user's document cannot be logged freely.
type APIError = providerhttp.APIError

// newClient applies the Hugging Face token fallback and hands the rest of the
// configuration to the shared transport.
//
// Resolving the credential stays here because the variable names are Hugging
// Face's own, and because every route this package calls requires one, which
// makes an unresolvable token an error rather than an anonymous request.
func newClient(cfg providerhttp.Config) (*providerhttp.Client, error) {
	if cfg.Token == "" {
		cfg.Token = os.Getenv("HF_TOKEN")
	}
	if cfg.Token == "" {
		cfg.Token = os.Getenv("HUGGING_FACE_HUB_TOKEN")
	}
	if cfg.Token == "" {
		return nil, errors.New("huggingface: token not set (use WithSharedToken or the HF_TOKEN env var)")
	}
	return providerhttp.New(providerName, cfg)
}
