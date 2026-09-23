// Codec tests run without a database: storage must not change command shape or
// normalize structured input bytes while carrying a message to the worker.
package postgres

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

func TestMailboxCodec(t *testing.T) {
	for _, block := range []sessionloop.InputBlock{
		{Kind: sessionloop.InputBlockText, Text: "plain text"},
		{Kind: sessionloop.InputBlockData, Data: json.RawMessage("null")},
		{Kind: sessionloop.InputBlockData, Data: json.RawMessage(" {\n  \"x\": 1.00 } ")},
	} {
		c := command("session", "one", "unused").Command
		c.Input.Blocks = []sessionloop.InputBlock{block}
		data, err := encodeCommand(c)
		if err != nil {
			t.Fatal(err)
		}
		var restored sessionloop.Command
		if err := decodeCommand(data, &restored); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(c, restored) {
			t.Fatalf("round trip changed command: %+v", restored)
		}
		if !bytes.Equal(block.Data, restored.Input.Blocks[0].Data) {
			t.Fatal("structured bytes changed")
		}
		if err := decodeCommand(append(data, 1), &restored); err == nil {
			t.Fatal("accepted trailing garbage")
		}
	}
}

func TestMailboxCodecCommandShapes(t *testing.T) {
	base := command("session", "one", "text").Command
	base.Input.Meta = map[string]string{}
	commands := []sessionloop.Command{base}
	for _, kind := range []sessionloop.CommandKind{sessionloop.CommandSteer, sessionloop.CommandFollowUp, sessionloop.CommandNextTurn} {
		c := base.Clone()
		c.Kind = kind
		if kind != sessionloop.CommandNextTurn {
			c.RunID = "run"
		}
		commands = append(commands, c)
	}
	commands = append(commands, sessionloop.Command{Kind: sessionloop.CommandInterrupt, RunID: "run"})
	for _, decisions := range [][]sessionloop.ResolutionDecision{nil, {}, {{ID: "decision", Action: sessionloop.ResolutionExternalResult, Data: json.RawMessage(" {\"result\": null} ")}}} {
		commands = append(commands, sessionloop.Command{Kind: sessionloop.CommandResolve, RunID: "run", Resolution: &sessionloop.Resolution{SuspensionID: "pause", Decisions: decisions}})
	}
	for _, c := range commands {
		data, err := encodeCommand(c)
		if err != nil {
			t.Fatal(err)
		}
		var restored sessionloop.Command
		if err := decodeCommand(data, &restored); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(c, restored) {
			t.Fatalf("%s changed shape: before=%#v after=%#v", c.Kind, c, restored)
		}
	}
}
