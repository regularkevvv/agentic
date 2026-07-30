package memory

import (
	"testing"

	"github.com/regularkevvv/agentic/harness/store"
	"github.com/regularkevvv/agentic/harness/store/storetest"
)

func TestRepositoryConformance(t *testing.T) {
	t.Parallel()
	storetest.Run(t, func(*testing.T) store.Repository { return New() })
}
