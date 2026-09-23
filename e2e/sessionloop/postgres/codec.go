// The versioned mailbox envelope carries structured Data as byte strings, not
// JSON values: absent data stays absent and JSON whitespace stays byte-exact.
// Data is moved out of Command while encoding, never stored in both places.

package postgres

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"

	"github.com/regularkevvv/agentic/harness/sessionloop"
)

type commandEnvelope struct {
	Version      int
	Command      sessionloop.Command
	InputData    [][]byte
	DecisionData [][]byte
}

func encodeCommand(c sessionloop.Command) ([]byte, error) {
	e := commandEnvelope{Version: 1, Command: c.Clone()}
	if e.Command.Input != nil {
		e.InputData = make([][]byte, len(e.Command.Input.Blocks))
		for i := range e.Command.Input.Blocks {
			e.InputData[i] = e.Command.Input.Blocks[i].Data
			e.Command.Input.Blocks[i].Data = nil
		}
	}
	if e.Command.Resolution != nil {
		e.DecisionData = make([][]byte, len(e.Command.Resolution.Decisions))
		for i := range e.Command.Resolution.Decisions {
			e.DecisionData[i] = e.Command.Resolution.Decisions[i].Data
			e.Command.Resolution.Decisions[i].Data = nil
		}
	}
	return json.Marshal(e)
}

func decodeCommand(data []byte, c *sessionloop.Command) error {
	var e commandEnvelope
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&e); err != nil {
		return err
	}
	if e.Version != 1 {
		return errors.New("postgres flavor: unknown mailbox envelope version")
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return errors.New("postgres flavor: trailing mailbox data")
	}
	if e.Command.Input != nil {
		if len(e.InputData) != len(e.Command.Input.Blocks) {
			return errors.New("postgres flavor: invalid input data envelope")
		}
		for i := range e.InputData {
			e.Command.Input.Blocks[i].Data = e.InputData[i]
		}
	} else if len(e.InputData) != 0 {
		return errors.New("postgres flavor: unexpected input data")
	}
	if e.Command.Resolution != nil {
		if len(e.DecisionData) != len(e.Command.Resolution.Decisions) {
			return errors.New("postgres flavor: invalid decision data envelope")
		}
		for i := range e.DecisionData {
			e.Command.Resolution.Decisions[i].Data = e.DecisionData[i]
		}
	} else if len(e.DecisionData) != 0 {
		return errors.New("postgres flavor: unexpected decision data")
	}
	*c = e.Command
	if err := c.Validate(); err != nil {
		return err
	}
	if c.Input != nil {
		return sessionloop.ValidateInput(*c.Input)
	}
	return nil
}
