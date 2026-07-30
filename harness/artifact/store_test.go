package artifact

import "testing"

func TestHandleValidation(t *testing.T) {
	t.Parallel()
	if err := ValidateHandle("art_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"); err != nil {
		t.Fatal(err)
	}
	if err := ValidateHandle("art_not-opaque"); err == nil {
		t.Fatal("invalid handle succeeded")
	}
}
