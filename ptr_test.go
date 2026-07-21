package agentic_test

import (
	"testing"

	"github.com/regularkevvv/agentic"
)

func TestPointerHelpersRoundTrip(t *testing.T) {
	t.Run("Int", func(t *testing.T) {
		// Zero must be representable as a set value, which is the whole reason
		// these fields are pointers.
		for _, want := range []int{0, 1024, -1} {
			if got := agentic.Int(want); got == nil || *got != want {
				t.Errorf("Int(%d) = %v, want a pointer to %d", want, got, want)
			}
		}
	})

	t.Run("Float64", func(t *testing.T) {
		for _, want := range []float64{0, 0.2, 1} {
			if got := agentic.Float64(want); got == nil || *got != want {
				t.Errorf("Float64(%v) = %v, want a pointer to %v", want, got, want)
			}
		}
	})

	t.Run("Bool", func(t *testing.T) {
		for _, want := range []bool{true, false} {
			if got := agentic.Bool(want); got == nil || *got != want {
				t.Errorf("Bool(%v) = %v, want a pointer to %v", want, got, want)
			}
		}
	})

	t.Run("String", func(t *testing.T) {
		for _, want := range []string{"", "stop"} {
			if got := agentic.String(want); got == nil || *got != want {
				t.Errorf("String(%q) = %v, want a pointer to %q", want, got, want)
			}
		}
	})

	t.Run("each call returns a distinct pointer", func(t *testing.T) {
		a, b := agentic.Int(1), agentic.Int(1)
		if a == b {
			t.Error("Int returned the same pointer twice; mutating one value would affect the other")
		}
	})
}
