// The old import path is an alias of the concrete local-channel implementation.
package memory

import "testing"

func TestCompatibilityConstructor(t *testing.T) {
	if New() == nil {
		t.Fatal("missing local-channel store")
	}
}
