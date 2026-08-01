// Package artifacts provides the gated, session-scoped read_artifact
// capability over opaque handles.
package artifacts

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

	agentic "github.com/regularkevvv/agentic"

	"github.com/regularkevvv/agentic/harness/artifact"
	"github.com/regularkevvv/agentic/harness/capability"
	"github.com/regularkevvv/agentic/harness/env"
	harnessruntime "github.com/regularkevvv/agentic/harness/runtime"
)

const (
	ID               = "artifacts"
	ToolReadArtifact = "read_artifact"
	DefaultReadBytes = 48 * 1024
)

type Config struct {
	Store        artifact.Store
	MaxReadBytes int
}

type Capability struct {
	store        artifact.Store
	maxReadBytes int
}

func New(config Config) (*Capability, error) {
	if config.Store == nil {
		return nil, errors.New("artifact capability store is required")
	}
	if config.MaxReadBytes == 0 {
		config.MaxReadBytes = DefaultReadBytes
	}
	if config.MaxReadBytes < 1 {
		return nil, errors.New("artifact read limit must be positive")
	}
	return &Capability{store: config.Store, maxReadBytes: config.MaxReadBytes}, nil
}

func (c *Capability) ID() string                    { return ID }
func (c *Capability) Ordering() capability.Ordering { return capability.Ordering{} }

func (c *Capability) Register(registry *capability.Registry) error {
	type input struct {
		Handle string `json:"handle" description:"Opaque artifact handle returned by a prior tool result"`
		Offset int    `json:"offset,omitempty" description:"Byte offset; defaults to zero"`
		Limit  int    `json:"limit,omitempty" description:"Bytes to read; bounded by the harness"`
	}
	type output struct {
		Handle     string `json:"handle"`
		Offset     int    `json:"offset"`
		NextOffset int    `json:"next_offset"`
		TotalBytes int    `json:"total_bytes"`
		EOF        bool   `json:"eof"`
		Content    string `json:"content"`
	}
	tool, handler, err := agentic.ToolWithContext(
		ToolReadArtifact,
		"Read a bounded chunk from a session-scoped artifact using only its opaque handle.",
		func(ctx context.Context, value input) (output, error) {
			runtime, ok := harnessruntime.FromContext(ctx)
			if !ok || runtime.SessionID == "" {
				return output{}, errors.New("read_artifact requires harness ToolRuntime")
			}
			handle := artifact.Handle(value.Handle)
			if err := artifact.ValidateHandle(handle); err != nil {
				return output{}, err
			}
			if value.Offset < 0 || value.Limit < 0 {
				return output{}, errors.New("artifact offset and limit cannot be negative")
			}
			data, err := c.store.Get(ctx, runtime.SessionID, handle)
			if err != nil {
				return output{}, err
			}
			if value.Offset > len(data) {
				return output{}, fmt.Errorf("artifact offset %d exceeds %d bytes", value.Offset, len(data))
			}
			if value.Offset < len(data) && !utf8.RuneStart(data[value.Offset]) {
				return output{}, errors.New("artifact offset must be a UTF-8 boundary")
			}
			limit := value.Limit
			if limit == 0 || limit > c.maxReadBytes {
				limit = c.maxReadBytes
			}
			end := value.Offset + limit
			if end > len(data) {
				end = len(data)
			}
			for end > value.Offset && !utf8.Valid(data[value.Offset:end]) {
				end--
			}
			return output{
				Handle:     handle.String(),
				Offset:     value.Offset,
				NextOffset: end,
				TotalBytes: len(data),
				EOF:        end == len(data),
				Content:    strings.ToValidUTF8(string(data[value.Offset:end]), "�"),
			}, nil
		},
	)
	if err != nil {
		return err
	}
	if err := registry.AddToolset(agentic.NewToolset().Add(tool, handler)); err != nil {
		return err
	}
	return registry.AddEffectResolver(ToolReadArtifact, capability.EffectResolverFunc(func(
		_ context.Context,
		call agentic.ToolUse,
		_ env.Environment,
	) (capability.Effect, error) {
		value, ok := call.Input["handle"].(string)
		handle := artifact.Handle(value)
		if !ok || artifact.ValidateHandle(handle) != nil {
			return capability.Effect{}, artifact.ErrInvalidHandle
		}
		return capability.Effect{
			Capability: "artifact",
			Action:     "read",
			Resource: env.CanonicalResource{
				Scheme:  "artifact",
				ID:      handle.String(),
				Display: handle.String(),
			},
		}, nil
	}))
}

var _ capability.Capability = (*Capability)(nil)
