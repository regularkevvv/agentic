package sessionloop_test

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

func validInput() *sessionloop.Input {
	return &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: "hello"}}}
}

func validResolution() *sessionloop.Resolution {
	return &sessionloop.Resolution{
		SuspensionID: "susp-1",
		Decisions: []sessionloop.ResolutionDecision{{
			ID:     "decision-1",
			Action: sessionloop.ResolutionApprove,
			Data:   json.RawMessage(`{"ok":true}`),
		}},
	}
}

func TestCommandValidateEnforcesTheStructuralMatrix(t *testing.T) {
	t.Parallel()
	valid := []sessionloop.Command{
		{Kind: sessionloop.CommandStart, Input: validInput()},
		{Kind: sessionloop.CommandSteer, RunID: "run-1", Input: validInput()},
		{Kind: sessionloop.CommandFollowUp, RunID: "run-1", Input: validInput()},
		{Kind: sessionloop.CommandNextTurn, Input: validInput()},
		{Kind: sessionloop.CommandResolve, RunID: "run-1", Resolution: validResolution()},
		{Kind: sessionloop.CommandResolve, RunID: "run-1", Input: validInput(), Resolution: validResolution()},
		{Kind: sessionloop.CommandInterrupt, RunID: "run-1"},
	}
	for _, command := range valid {
		if err := command.Validate(); err != nil {
			t.Errorf("Validate(%s) = %v, want nil", command.Kind, err)
		}
	}

	invalid := []struct {
		name    string
		command sessionloop.Command
	}{
		{"start with run target", sessionloop.Command{Kind: sessionloop.CommandStart, RunID: "run-1", Input: validInput()}},
		{"start without input", sessionloop.Command{Kind: sessionloop.CommandStart}},
		{"start with resolution", sessionloop.Command{Kind: sessionloop.CommandStart, Input: validInput(), Resolution: validResolution()}},
		{"steer without run target", sessionloop.Command{Kind: sessionloop.CommandSteer, Input: validInput()}},
		{"steer without input", sessionloop.Command{Kind: sessionloop.CommandSteer, RunID: "run-1"}},
		{"steer with resolution", sessionloop.Command{Kind: sessionloop.CommandSteer, RunID: "run-1", Input: validInput(), Resolution: validResolution()}},
		{"follow-up without run target", sessionloop.Command{Kind: sessionloop.CommandFollowUp, Input: validInput()}},
		{"follow-up without input", sessionloop.Command{Kind: sessionloop.CommandFollowUp, RunID: "run-1"}},
		{"follow-up with resolution", sessionloop.Command{Kind: sessionloop.CommandFollowUp, RunID: "run-1", Input: validInput(), Resolution: validResolution()}},
		{"next-turn with run target", sessionloop.Command{Kind: sessionloop.CommandNextTurn, RunID: "run-1", Input: validInput()}},
		{"next-turn without input", sessionloop.Command{Kind: sessionloop.CommandNextTurn}},
		{"next-turn with resolution", sessionloop.Command{Kind: sessionloop.CommandNextTurn, Input: validInput(), Resolution: validResolution()}},
		{"resolve without run target", sessionloop.Command{Kind: sessionloop.CommandResolve, Resolution: validResolution()}},
		{"resolve without resolution", sessionloop.Command{Kind: sessionloop.CommandResolve, RunID: "run-1"}},
		{"interrupt without run target", sessionloop.Command{Kind: sessionloop.CommandInterrupt}},
		{"interrupt with input", sessionloop.Command{Kind: sessionloop.CommandInterrupt, RunID: "run-1", Input: validInput()}},
		{"interrupt with resolution", sessionloop.Command{Kind: sessionloop.CommandInterrupt, RunID: "run-1", Resolution: validResolution()}},
		{"unknown kind", sessionloop.Command{Kind: "teleport"}},
	}
	for _, violation := range invalid {
		if err := violation.command.Validate(); !errors.Is(err, sessionloop.ErrInvalidCommand) {
			t.Errorf("%s: Validate() = %v, want ErrInvalidCommand", violation.name, err)
		}
	}
}

func TestCommandValidateChecksResolutionStructure(t *testing.T) {
	t.Parallel()
	base := func(resolution *sessionloop.Resolution) sessionloop.Command {
		return sessionloop.Command{Kind: sessionloop.CommandResolve, RunID: "run-1", Resolution: resolution}
	}
	invalid := []struct {
		name       string
		resolution *sessionloop.Resolution
	}{
		{"missing suspension ID", &sessionloop.Resolution{}},
		{"decision without ID", &sessionloop.Resolution{SuspensionID: "susp-1", Decisions: []sessionloop.ResolutionDecision{{Action: sessionloop.ResolutionApprove}}}},
		{"decision with unknown action", &sessionloop.Resolution{SuspensionID: "susp-1", Decisions: []sessionloop.ResolutionDecision{{ID: "d", Action: "shrug"}}}},
		{"decision with truncated JSON data", &sessionloop.Resolution{SuspensionID: "susp-1", Decisions: []sessionloop.ResolutionDecision{{ID: "d", Action: sessionloop.ResolutionExternalResult, Data: json.RawMessage(`{"partial"`)}}}},
	}
	for _, violation := range invalid {
		if err := base(violation.resolution).Validate(); !errors.Is(err, sessionloop.ErrInvalidCommand) {
			t.Errorf("%s: Validate() = %v, want ErrInvalidCommand", violation.name, err)
		}
	}
	deny := base(&sessionloop.Resolution{SuspensionID: "susp-1", Decisions: []sessionloop.ResolutionDecision{{ID: "d", Action: sessionloop.ResolutionDeny, Reason: "not now"}}})
	if err := deny.Validate(); err != nil {
		t.Fatalf("a deny decision without data must validate, got %v", err)
	}
}

func TestValidateCommandScreensCapabilitiesAfterStructure(t *testing.T) {
	t.Parallel()
	none := sessionloop.NewCapabilities()
	all := sessionloop.NewCapabilities(
		sessionloop.CapabilitySteer,
		sessionloop.CapabilityFollowUp,
		sessionloop.CapabilityNextTurn,
		sessionloop.CapabilityInterrupt,
		sessionloop.CapabilitySuspensionResolve,
		sessionloop.CapabilityIdempotentDispatch,
	)

	if err := sessionloop.ValidateCommand(sessionloop.Command{Kind: sessionloop.CommandStart, Input: validInput()}, none); err != nil {
		t.Fatalf("start must always be allowed as a baseline requirement, got %v", err)
	}

	unsupported := []sessionloop.Command{
		{Kind: sessionloop.CommandSteer, RunID: "run-1", Input: validInput()},
		{Kind: sessionloop.CommandFollowUp, RunID: "run-1", Input: validInput()},
		{Kind: sessionloop.CommandNextTurn, Input: validInput()},
		{Kind: sessionloop.CommandInterrupt, RunID: "run-1"},
		{Kind: sessionloop.CommandResolve, RunID: "run-1", Resolution: validResolution()},
		{Kind: sessionloop.CommandStart, Input: validInput(), IdempotencyKey: "key-1"},
	}
	for _, command := range unsupported {
		if err := sessionloop.ValidateCommand(command, none); !errors.Is(err, sessionloop.ErrUnsupported) {
			t.Errorf("ValidateCommand(%s) without capabilities = %v, want ErrUnsupported", command.Kind, err)
		}
		if err := sessionloop.ValidateCommand(command, all); err != nil {
			t.Errorf("ValidateCommand(%s) with capabilities = %v, want nil", command.Kind, err)
		}
	}

	structurallyBroken := sessionloop.Command{Kind: sessionloop.CommandSteer, Input: validInput(), IdempotencyKey: "key-1"}
	if err := sessionloop.ValidateCommand(structurallyBroken, none); !errors.Is(err, sessionloop.ErrInvalidCommand) {
		t.Fatalf("structural validation must precede capability screening, got %v", err)
	}
}

func TestCommandCloneIsDeeplyIndependent(t *testing.T) {
	t.Parallel()
	original := sessionloop.Command{
		ID:         "cmd-1",
		Kind:       sessionloop.CommandResolve,
		RunID:      "run-1",
		Input:      &sessionloop.Input{Blocks: []sessionloop.InputBlock{{Kind: sessionloop.InputBlockText, Text: "keep"}}, Meta: map[string]string{"k": "v"}},
		Resolution: validResolution(),
	}
	clone := original.Clone()
	clone.Input.Blocks[0].Text = "mutated"
	clone.Input.Meta["k"] = "mutated"
	clone.Resolution.SuspensionID = "mutated"
	clone.Resolution.Decisions[0].Data[2] = 'X'
	if original.Input.Blocks[0].Text != "keep" || original.Input.Meta["k"] != "v" {
		t.Fatalf("mutating the cloned input leaked into the original: %#v", original.Input)
	}
	if original.Resolution.SuspensionID != "susp-1" || string(original.Resolution.Decisions[0].Data) != `{"ok":true}` {
		t.Fatalf("mutating the cloned resolution leaked into the original: %#v", original.Resolution)
	}

	bare := sessionloop.Command{Kind: sessionloop.CommandInterrupt, RunID: "run-1"}
	cloneBare := bare.Clone()
	if cloneBare.Input != nil || cloneBare.Resolution != nil {
		t.Fatalf("cloning a command without payloads invented payloads: %#v", cloneBare)
	}
}
