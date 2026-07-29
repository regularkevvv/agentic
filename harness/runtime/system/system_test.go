package system

import (
	"strings"
	"testing"
	"time"
)

func TestAdaptersSupplyTimeAndOpaqueIdentifiers(t *testing.T) {
	t.Parallel()
	before := time.Now()
	observed := NewClock().Now()
	after := time.Now()
	if observed.Before(before) || observed.After(after) {
		t.Fatalf("clock returned %s outside [%s, %s]", observed, before, after)
	}

	first, err := NewIDs().New("session")
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewIDs().New("session")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(first, "session_") || first == second {
		t.Fatalf("IDs = %q, %q", first, second)
	}
	if _, err := NewIDs().New(""); err == nil {
		t.Fatal("empty prefix succeeded")
	}
}
