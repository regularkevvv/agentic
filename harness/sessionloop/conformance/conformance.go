package conformance

import (
	"testing"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

// Scenario metadata keys and values of the Factory contract. A start input
// whose Meta maps MetaScenario to one of the Scenario values asks the host
// to exhibit the documented optional behavior.
const (
	MetaScenario    = "conformance.scenario"
	ScenarioPreview = "preview"
	ScenarioTools   = "tools"
	ScenarioSuspend = "suspend"
	ScenarioOutput  = "output"
)

// Gate is the only black-box control hook the suite uses for deterministic
// scheduling: HoldNextRun makes the next started run block before its first
// step until the returned release function is called. Release must be
// idempotent.
type Gate interface {
	HoldNextRun() (release func())
}

// Env is one fresh conformance environment. Cleanup, when non-nil, runs
// after the case. A nil Gate skips timing-dependent cases.
type Env struct {
	Host    sessionloop.Host
	Gate    Gate
	Cleanup func()
}

// Factory produces a fresh Env per case. See the package documentation for
// the behavioral contract the returned host must honor.
type Factory func(t *testing.T) Env

type conformanceCase struct {
	name      string
	requires  []sessionloop.Capability
	needsGate bool
	run       func(t *testing.T, env Env, capabilities sessionloop.Capabilities)
}

// Run executes every baseline case and every optional case whose
// capabilities the host advertises against a fresh Env from the factory.
func Run(t *testing.T, factory Factory) {
	t.Helper()
	if factory == nil {
		t.Fatal("conformance: factory must not be nil")
	}
	for _, current := range cases() {
		t.Run(current.name, func(t *testing.T) {
			env := factory(t)
			if env.Host == nil {
				t.Fatal("conformance: factory returned a nil host")
			}
			if env.Cleanup != nil {
				t.Cleanup(env.Cleanup)
			}
			capabilities := probeCapabilities(t, env.Host)
			for _, capability := range current.requires {
				if !capabilities.Supports(capability) {
					t.Skipf("host does not advertise capability %q", capability)
				}
			}
			if current.needsGate && env.Gate == nil {
				t.Skip("factory provided no Gate; timing-dependent case skipped")
			}
			current.run(t, env, capabilities)
		})
	}
}

func cases() []conformanceCase {
	return []conformanceCase{
		{
			name: "start receipt carries the exact run identity observed in run.started and run.settled",
			run:  caseStartReceiptRunIdentity,
		},
		{
			name: "authoritative entries order the start user entry before assistant output with settlement last",
			run:  caseAuthoritativeEntryOrder,
		},
		{
			name: "each run settles exactly once across consecutive runs",
			run:  caseOneSettlementPerRun,
		},
		{
			name: "the session is idle after settlement and accepts a second run",
			run:  caseIdleAfterSettlement,
		},
		{
			name: "snapshots are copy owned",
			run:  caseSnapshotCopyOwnership,
		},
		{
			name: "a canceled Next reports the context error and closing a stream mid wait ends it with EOF",
			run:  caseStreamCloseAndCanceledNext,
		},
		{
			name: "every structural violation of the command matrix is rejected with ErrInvalidCommand",
			run:  caseInvalidCommandMatrix,
		},
		{
			name: "closing a session is idempotent and a closed session can be reopened",
			run:  caseCloseAndReopenLifecycle,
		},
		{
			name:      "targeted commands naming a settled run are rejected with ErrStaleRun",
			needsGate: true,
			run:       caseStaleTargetedCommand,
		},
		{
			name:      "concurrent starts admit exactly one run and reject the other with ErrSessionBusy",
			needsGate: true,
			run:       caseConcurrentStartSingleFlight,
		},
		{
			name: "no events follow close except stream termination",
			run:  caseNoEventsAfterClose,
		},
		{
			name:     "durable receipts have positions and replay from a mid stream position yields an identical suffix",
			requires: []sessionloop.Capability{sessionloop.CapabilityReplay},
			run:      caseDurableReplay,
		},
		{
			name:     "previews arrive between authoritative events and never alter committed entries",
			requires: []sessionloop.Capability{sessionloop.CapabilityPreview},
			run:      casePreview,
		},
		{
			name:      "a steer accepted mid run commits an entry with origin steer before settlement",
			requires:  []sessionloop.Capability{sessionloop.CapabilitySteer},
			needsGate: true,
			run:       caseSteerMidRun,
		},
		{
			name:      "a follow-up accepted mid run commits an entry with origin follow_up before settlement",
			requires:  []sessionloop.Capability{sessionloop.CapabilityFollowUp},
			needsGate: true,
			run:       caseFollowUpMidRun,
		},
		{
			name:      "a next-turn queued while busy survives interruption and drains into the next run",
			requires:  []sessionloop.Capability{sessionloop.CapabilityNextTurn},
			needsGate: true,
			run:       caseNextTurnSurvivesInterrupt,
		},
		{
			name:      "an interrupt settles the active run as interrupted",
			requires:  []sessionloop.Capability{sessionloop.CapabilityInterrupt},
			needsGate: true,
			run:       caseInterrupt,
		},
		{
			name:     "a suspension resolves the same run to settlement and rejects a wrong suspension ID without consuming it",
			requires: []sessionloop.Capability{sessionloop.CapabilitySuspensionResolve},
			run:      caseSuspensionResolve,
		},
		{
			name:     "a suspension survives close and reopen and still resolves the same run to settlement",
			requires: []sessionloop.Capability{sessionloop.CapabilitySuspensionResolve},
			run:      caseSuspensionSurvivesCloseAndReopen,
		},
		{
			name:     "detailed tool content carries tool_call and tool_result blocks with valid JSON data",
			requires: []sessionloop.Capability{sessionloop.CapabilityDetailedTools},
			run:      caseDetailedTools,
		},
		{
			name:     "structured output is present on the completed outcome",
			requires: []sessionloop.Capability{sessionloop.CapabilityStructuredOutput},
			run:      caseStructuredOutput,
		},
		{
			name:     "dispatching the same idempotency key twice returns the same receipt identity without duplicate effects",
			requires: []sessionloop.Capability{sessionloop.CapabilityIdempotentDispatch},
			run:      caseIdempotentDispatch,
		},
	}
}
