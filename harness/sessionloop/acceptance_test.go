// Rejection reason validation keeps terminal resolution categories explicit.
package sessionloop

import "testing"

func TestRejectionReasons(t *testing.T) {
	for _, reason := range []Rejection{RejectionStaleRun, RejectionNotRunning, RejectionUnsupported, RejectionInvalidCommand} {
		if !reason.Valid() {
			t.Fatal(reason)
		}
	}
	for _, reason := range []Rejection{"", "timeout", "storage_error"} {
		if reason.Valid() {
			t.Fatal(reason)
		}
	}
}
