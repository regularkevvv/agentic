package sagemaker

import (
	"encoding/json"
	"fmt"

	"github.com/regularkevvv/agentic/internal/retrieval"
	"github.com/regularkevvv/agentic/internal/retrieval/wire"
)

// marshalRequest encodes the agentic.representations.v1 body sent to the
// endpoint.
func marshalRequest(req *retrieval.RepresentationRequest) ([]byte, error) {
	body, err := json.Marshal(wire.NewRequest(req))
	if err != nil {
		return nil, fmt.Errorf("sagemaker: encode request: %w", err)
	}
	return body, nil
}
