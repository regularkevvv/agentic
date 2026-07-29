package memory

import (
	"testing"

	"github.com/regularkevvv/agentic/harness/artifact"
	"github.com/regularkevvv/agentic/harness/artifact/artifacttest"
)

func TestStoreConformance(t *testing.T) {
	t.Parallel()
	artifacttest.Run(t, func(*testing.T) artifact.Store { return New() })
}
