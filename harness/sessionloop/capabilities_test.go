package sessionloop_test

import (
	"reflect"
	"testing"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

func TestNewCapabilitiesDeduplicatesSortsAndPreservesUnknownStrings(t *testing.T) {
	t.Parallel()
	capabilities := sessionloop.NewCapabilities(
		sessionloop.CapabilitySteer,
		sessionloop.Capability("x.vendor.extension"),
		sessionloop.CapabilityReplay,
		sessionloop.CapabilitySteer,
		sessionloop.CapabilityReplay,
	)
	want := sessionloop.Capabilities{
		sessionloop.CapabilityReplay,
		sessionloop.CapabilitySteer,
		sessionloop.Capability("x.vendor.extension"),
	}
	if !reflect.DeepEqual(capabilities, want) {
		t.Fatalf("NewCapabilities = %#v, want deduplicated sorted %#v", capabilities, want)
	}
	if !capabilities.Supports("x.vendor.extension") {
		t.Fatal("an unknown capability string must survive the round trip and stay queryable")
	}
	if capabilities.Supports(sessionloop.CapabilityInterrupt) {
		t.Fatal("Supports reported a capability that was never advertised")
	}
}

func TestNewCapabilitiesCopiesItsArguments(t *testing.T) {
	t.Parallel()
	source := []sessionloop.Capability{sessionloop.CapabilitySteer, sessionloop.CapabilityReplay}
	capabilities := sessionloop.NewCapabilities(source...)
	source[0] = "mutated"
	if capabilities.Supports("mutated") || !capabilities.Supports(sessionloop.CapabilitySteer) {
		t.Fatalf("mutating the argument slice leaked into the set: %#v", capabilities)
	}
}

func TestCapabilitiesCloneIsIndependentAndNilStaysNil(t *testing.T) {
	t.Parallel()
	original := sessionloop.NewCapabilities(sessionloop.CapabilitySteer, sessionloop.CapabilityReplay)
	clone := original.Clone()
	clone[0] = "mutated"
	if original[0] == "mutated" {
		t.Fatalf("mutating the clone leaked into the original: %#v", original)
	}
	var absent sessionloop.Capabilities
	if absent.Clone() != nil {
		t.Fatal("cloning a nil capability set must stay nil")
	}
}

func TestSessionOptionsCloneCopiesMetadata(t *testing.T) {
	t.Parallel()
	original := sessionloop.SessionOptions{Meta: map[string]string{"tenant": "acme"}}
	clone := original.Clone()
	clone.Meta["tenant"] = "mutated"
	if original.Meta["tenant"] != "acme" {
		t.Fatalf("mutating the clone leaked into the original: %#v", original.Meta)
	}
	if empty := (sessionloop.SessionOptions{}).Clone(); empty.Meta != nil {
		t.Fatalf("cloning empty options invented metadata: %#v", empty.Meta)
	}
}

func TestPositionZeroMeansNotReplayableAndEqualityIsFieldwise(t *testing.T) {
	t.Parallel()
	var zero sessionloop.Position
	if !zero.IsZero() {
		t.Fatal("the zero position must report IsZero")
	}
	if (sessionloop.Position{Sequence: 1}).IsZero() || (sessionloop.Position{Token: "tk-1"}).IsZero() {
		t.Fatal("a position with any field set is replayable and must not report IsZero")
	}
	left := sessionloop.Position{Sequence: 7, Token: "tk-7"}
	if !left.Equal(sessionloop.Position{Sequence: 7, Token: "tk-7"}) {
		t.Fatal("identical positions must compare equal")
	}
	if left.Equal(sessionloop.Position{Sequence: 7, Token: "other"}) || left.Equal(sessionloop.Position{Sequence: 8, Token: "tk-7"}) {
		t.Fatal("positions differing in any field must not compare equal")
	}
}
