package sessionloop_test

import (
	"testing"

	"github.com/regularkevvv/agentic/harness/sessionloop/conformance"
	"github.com/regularkevvv/agentic/harness/sessionloop/testkit"
)

// TestTestkitReferenceHostPassesTheConformanceSuite exercises both companion
// packages from the module's own tests: the conformance suite runs every
// baseline case and every optional case the testkit host advertises,
// including the timing-dependent cases via the host's HoldNextRun gate.
func TestTestkitReferenceHostPassesTheConformanceSuite(t *testing.T) {
	conformance.Run(t, func(*testing.T) conformance.Env {
		host := testkit.New(testkit.WithRunFunc(testkit.ScenarioRunFunc()))
		return conformance.Env{Host: host, Gate: host}
	})
}

// TestTestkitIdempotentHostPassesTheConformanceSuite runs the suite against
// a testkit host that opts into dispatch.idempotent, so the idempotency case
// executes against a real key map instead of skipping everywhere.
func TestTestkitIdempotentHostPassesTheConformanceSuite(t *testing.T) {
	conformance.Run(t, func(*testing.T) conformance.Env {
		host := testkit.New(testkit.WithRunFunc(testkit.ScenarioRunFunc()), testkit.WithIdempotentDispatch())
		return conformance.Env{Host: host, Gate: host}
	})
}

// TestConformanceSkipsTimingDependentCasesWithoutAGate documents the Factory
// contract for hosts that cannot provide deterministic scheduling: with a
// nil Gate the timing-dependent cases skip instead of flaking, and every
// other case still runs.
func TestConformanceSkipsTimingDependentCasesWithoutAGate(t *testing.T) {
	cleaned := false
	conformance.Run(t, func(*testing.T) conformance.Env {
		return conformance.Env{
			Host:    testkit.New(testkit.WithRunFunc(testkit.ScenarioRunFunc())),
			Cleanup: func() { cleaned = true },
		}
	})
	if !cleaned {
		t.Fatal("conformance.Run never invoked the Env cleanup")
	}
}
