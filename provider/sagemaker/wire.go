package sagemaker

import (
	"encoding/json"
	"fmt"

	"github.com/regularkevvv/agentic/internal/core"
	"github.com/regularkevvv/agentic/internal/representationwire"
)

// marshalRequest encodes the agentic.representations.v1 body sent to the
// endpoint.
func marshalRequest(req *core.RepresentationRequest) ([]byte, error) {
	body, err := json.Marshal(representationwire.NewRequest(req))
	if err != nil {
		return nil, fmt.Errorf("sagemaker: encode request: %w", err)
	}
	return body, nil
}
