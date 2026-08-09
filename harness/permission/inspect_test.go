package permission

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/env"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

func permissionTestSuspension(t *testing.T, required []string, requests []deferredRequest) agentic.Suspension {
	t.Helper()
	requiredSet := make(map[string]bool, len(required))
	for _, id := range required {
		requiredSet[id] = true
	}
	preAllowed := make([]string, 0, 2-len(required))
	for _, id := range []string{"two", "one"} {
		if !requiredSet[id] {
			preAllowed = append(preAllowed, id)
		}
	}
	inner, err := json.Marshal(deferralPayload{DeferredBatch: harnessruntime.DeferredBatch{
		Version: 1, RequiredResolutionIDs: required, PreAllowedCallIDs: preAllowed,
	}, Requests: requests})
	if err != nil {
		t.Fatal(err)
	}
	id := "permission-suspension"
	root, err := json.Marshal(map[string]any{
		"Version": 1, "SuspensionID": id,
		"Calls":             []agentic.ToolUse{{ID: "two", Name: "second"}, {ID: "one", Name: "first"}},
		"ExecutableCallIDs": []string{"two", "one"},
		"Deferral":          agentic.ToolDeferral{Kind: harnessruntime.PermissionDeferralKind, Payload: inner},
	})
	if err != nil {
		t.Fatal(err)
	}
	return agentic.Suspension{ID: id, Kind: harnessruntime.PermissionDeferralKind, Payload: root}
}

func request(id string) deferredRequest {
	return deferredRequest{CallID: id, Request: PermissionRequest{
		Capability: "filesystem", Action: "write",
		CanonicalResource: env.CanonicalResource{Scheme: "file", ID: "/" + id, Display: id},
	}}
}

func TestInspectSuspensionReturnsExecutableOrder(t *testing.T) {
	t.Parallel()
	suspension := permissionTestSuspension(t, []string{"one", "two"}, []deferredRequest{request("one"), request("two")})
	approvals, err := InspectSuspension(suspension)
	if err != nil {
		t.Fatal(err)
	}
	got := []string{approvals[0].CallID, approvals[0].ToolName, approvals[1].CallID, approvals[1].ToolName}
	want := []string{"two", "second", "one", "first"}
	if !reflect.DeepEqual(got, want) || approvals[0].Request.CanonicalResource.Display != "two" {
		t.Fatalf("approvals = %#v", approvals)
	}
}

func TestInspectSuspensionRejectsMalformedApprovalViews(t *testing.T) {
	t.Parallel()
	cases := []agentic.Suspension{
		{ID: "other", Kind: "other"},
		permissionTestSuspension(t, []string{"one"}, nil),
		permissionTestSuspension(t, []string{"one"}, []deferredRequest{{CallID: "unknown", Request: request("one").Request}}),
		permissionTestSuspension(t, []string{"one"}, []deferredRequest{request("one"), request("one")}),
		permissionTestSuspension(t, []string{"one"}, []deferredRequest{{CallID: "one", Request: PermissionRequest{}}}),
	}
	for index, value := range cases {
		if _, err := InspectSuspension(value); err == nil {
			t.Fatalf("case %d succeeded", index)
		}
	}

	malformed := permissionTestSuspension(t, []string{"one"}, []deferredRequest{request("one")})
	var root map[string]any
	if err := json.Unmarshal(malformed.Payload, &root); err != nil {
		t.Fatal(err)
	}
	root["Deferral"] = agentic.ToolDeferral{Kind: harnessruntime.PermissionDeferralKind, Payload: []byte("bad")}
	malformed.Payload, _ = json.Marshal(root)
	if _, err := InspectSuspension(malformed); !errors.Is(err, harnessruntime.ErrInvalidResumeRequest) {
		t.Fatalf("malformed payload error = %v", err)
	}

	malformedRequests := permissionTestSuspension(t, []string{"one", "two"}, []deferredRequest{request("one"), request("two")})
	if err := json.Unmarshal(malformedRequests.Payload, &root); err != nil {
		t.Fatal(err)
	}
	root["Deferral"] = agentic.ToolDeferral{
		Kind:    harnessruntime.PermissionDeferralKind,
		Payload: []byte(`{"version":1,"required_resolution_ids":["one","two"],"requests":"bad"}`),
	}
	malformedRequests.Payload, _ = json.Marshal(root)
	if _, err := InspectSuspension(malformedRequests); !errors.Is(err, harnessruntime.ErrInvalidResumeRequest) {
		t.Fatalf("malformed requests error = %v", err)
	}
}
