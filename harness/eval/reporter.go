package eval

import (
	"encoding/json"
	"errors"
	"io"
)

func WriteJSON[I, O any](writer io.Writer, report Report[I, O]) error {
	if writer == nil {
		return errors.New("eval JSON writer is required")
	}
	if report.Version != 1 {
		return errors.New("unsupported eval report version")
	}
	encoder := json.NewEncoder(writer)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(report)
}
