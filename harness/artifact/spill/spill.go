package spill

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness/artifact"
)

const (
	DefaultThreshold = 64 * 1024
	DefaultHead      = 24 * 1024
	DefaultTail      = 24 * 1024
)

type Config struct {
	Threshold int
	Head      int
	Tail      int
	// Disabled is the explicit opt-out from result bounding. Zero-valued
	// configuration keeps the safe default enabled.
	Disabled bool
}

// Validate checks a spill configuration without opening storage.
func (c Config) Validate() error {
	_, err := c.normalized()
	return err
}

func (c Config) normalized() (Config, error) {
	if c.Disabled {
		if c.Threshold != 0 || c.Head != 0 || c.Tail != 0 {
			return Config{}, errors.New("disabled artifact spill cannot set limits")
		}
		return c, nil
	}
	if c.Threshold == 0 {
		c.Threshold = DefaultThreshold
	}
	if c.Head == 0 {
		c.Head = DefaultHead
	}
	if c.Tail == 0 {
		c.Tail = DefaultTail
	}
	if c.Threshold < 1 || c.Head < 0 || c.Tail < 0 || c.Head+c.Tail > c.Threshold {
		return Config{}, errors.New("invalid artifact spill limits")
	}
	return c, nil
}

// Processor performs Agentic's one result-formatting pass, spills oversized
// bytes once per call ID, and replaces only the model-visible content.
type Processor struct {
	store     artifact.Store
	sessionID string
	config    Config
}

func NewProcessor(storage artifact.Store, sessionID string, config Config) (*Processor, error) {
	if storage == nil {
		return nil, errors.New("artifact store is required")
	}
	if err := artifact.ValidateSessionID(sessionID); err != nil {
		return nil, err
	}
	normalized, err := config.normalized()
	if err != nil {
		return nil, err
	}
	return &Processor{store: storage, sessionID: sessionID, config: normalized}, nil
}

type Factory struct {
	store  artifact.Store
	config Config
}

func NewFactory(storage artifact.Store, config Config) (*Factory, error) {
	if storage == nil {
		return nil, errors.New("artifact store is required")
	}
	normalized, err := config.normalized()
	if err != nil {
		return nil, err
	}
	return &Factory{store: storage, config: normalized}, nil
}

func (f *Factory) Open(ctx context.Context, sessionID string) (agentic.ToolResultProcessor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return NewProcessor(f.store, sessionID, f.config)
}

func (p *Processor) Process(ctx context.Context, call agentic.ToolUse, result agentic.ToolExecutionResult) (agentic.ToolExecutionResult, error) {
	formatted := strings.ToValidUTF8(agentic.FormatToolResult(result.Content), "�")
	data := []byte(formatted)
	result.Content = formatted
	if p.config.Disabled {
		return result, nil
	}
	if len(data) <= p.config.Threshold {
		return result, nil
	}
	handle, err := p.store.Put(ctx, p.sessionID, call.ID, data)
	if err != nil {
		return result, fmt.Errorf("spill tool result: %w", err)
	}
	head := utf8Head(data, p.config.Head)
	tail := utf8Tail(data, p.config.Tail)
	result.Content = fmt.Sprintf(
		"[harness artifact %s; full_bytes=%d; shown_head=%d; shown_tail=%d]\n%s\n…\n%s",
		handle, len(data), len(head), len(tail), head, tail,
	)
	return result, nil
}

var _ artifact.ProcessorFactory = (*Factory)(nil)
var _ agentic.ToolResultProcessor = (*Processor)(nil)

func utf8Head(data []byte, limit int) string {
	if limit >= len(data) {
		return string(data)
	}
	end := limit
	for end > 0 && !utf8.Valid(data[:end]) {
		end--
	}
	return string(data[:end])
}

func utf8Tail(data []byte, limit int) string {
	if limit >= len(data) {
		return string(data)
	}
	start := len(data) - limit
	for start < len(data) && !utf8.Valid(data[start:]) {
		start++
	}
	return string(data[start:])
}
