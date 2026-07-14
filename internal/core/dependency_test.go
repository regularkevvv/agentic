package core

import (
	"strings"
	"testing"
	"unsafe"
)

func TestDependencyEnvelope(t *testing.T) {
	type deps struct{ Value string }
	envelope := NewDependencyEnvelope(deps{Value: "ok"})
	if envelope.DependencyType() == nil || envelope.DependencyIsNil() {
		t.Fatalf("unexpected envelope state: type=%v nil=%v", envelope.DependencyType(), envelope.DependencyIsNil())
	}
	got, err := ExtractDependency[deps](envelope)
	if err != nil || got.Value != "ok" {
		t.Fatalf("extract: got %#v, err %v", got, err)
	}
}

func TestDependencyEnvelopeTypedNilKinds(t *testing.T) {
	var pointer *int
	var values map[string]int
	var items []int
	var fn func()
	var ch chan int
	var iface any
	var unsafePointer = (*byte)(nil)
	var rawUnsafePointer unsafe.Pointer

	cases := []DependencyEnvelope{
		NewDependencyEnvelope(pointer),
		NewDependencyEnvelope(values),
		NewDependencyEnvelope(items),
		NewDependencyEnvelope(fn),
		NewDependencyEnvelope(ch),
		NewDependencyEnvelope(iface),
		NewDependencyEnvelope(unsafePointer),
		NewDependencyEnvelope(rawUnsafePointer),
	}
	for i, envelope := range cases {
		if !envelope.DependencyIsNil() {
			t.Errorf("case %d: expected nil", i)
		}
	}
	if NewDependencyEnvelope(struct{}{}).DependencyIsNil() {
		t.Fatal("zero struct must not be treated as nil")
	}
}

func TestExtractDependencyErrors(t *testing.T) {
	if _, err := ExtractDependency[int]("wrong"); err == nil || !strings.Contains(err.Error(), "envelope") {
		t.Fatalf("expected envelope error, got %v", err)
	}
	if _, err := ExtractDependency[int](NewDependencyEnvelope("wrong")); err == nil || !strings.Contains(err.Error(), "type") {
		t.Fatalf("expected type error, got %v", err)
	}
}
