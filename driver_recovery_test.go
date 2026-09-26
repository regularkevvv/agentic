package agentic

// Recovery is explicit: ordinary continuation still rejects a final assistant,
// while a committed candidate can be validated without regenerating its bytes.
import (
	"context"
	"errors"
	"reflect"
	"testing"

	providertest "github.com/regularkevvv/agentic/provider/test"
)

func TestDriveRecoverCommittedCandidate(t *testing.T) {
	history := []Message{NewTextMessage(RoleUser, "input"), NewTextMessage(RoleAssistant, "committed")}
	model := providertest.NewTestModel()
	validated := 0
	a := NewAgent("", model, WithOutputValidatorFunc(func(_ context.Context, output string) error {
		validated++
		if output != "committed" {
			t.Errorf("candidate=%q", output)
		}
		return nil
	}))
	execution, err := a.Drive(t.Context(), DriveInput{Mode: DriveRecover, History: history})
	if err != nil {
		t.Fatal(err)
	}
	if execution.Status != ExecutionCompleted || execution.Result.Output != "committed" || validated != 1 {
		t.Fatalf("execution=%+v validations=%d", execution, validated)
	}
	if len(model.Calls()) != 0 || !reflect.DeepEqual(history, execution.Result.Messages) {
		t.Fatal("recovery regenerated or changed the committed candidate")
	}
	if _, err := a.Drive(t.Context(), DriveInput{Mode: DriveContinue, History: history}); !errors.Is(err, ErrDriveInput) {
		t.Fatalf("ordinary continuation accidentally relaxed: %v", err)
	}
}

func TestDriveRecoverValidationRetryAndOpenFrontier(t *testing.T) {
	model := providertest.NewTestModel(providertest.ModelResponse{Text: "corrected"})
	validated := 0
	a := NewAgent("", model, WithOutputValidatorFunc(func(_ context.Context, output string) error {
		validated++
		if output == "bad candidate" {
			return errors.New("correct the candidate")
		}
		return nil
	}))
	history := []Message{NewTextMessage(RoleUser, "input"), NewTextMessage(RoleAssistant, "bad candidate")}
	execution, err := a.Drive(t.Context(), DriveInput{Mode: DriveRecover, History: history})
	if err != nil {
		t.Fatal(err)
	}
	if execution.Status != ExecutionCompleted || execution.Result.Output != "corrected" || validated != 2 || len(model.Calls()) != 1 {
		t.Fatalf("execution=%+v validations=%d calls=%d", execution, validated, len(model.Calls()))
	}
	for i, original := range history {
		if !reflect.DeepEqual(original, execution.Result.Messages[i]) {
			t.Fatal("validation retry rewrote history")
		}
	}
	open := []Message{NewTextMessage(RoleUser, "input"), NewToolUseMessage(ToolUse{ID: "call", Name: "read", Input: map[string]any{}})}
	if _, err := a.Drive(t.Context(), DriveInput{Mode: DriveRecover, History: open}); !errors.Is(err, ErrTranscriptInvalid) {
		t.Fatalf("unpaired tools require repair/resolution: %v", err)
	}
}
