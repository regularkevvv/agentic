package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/coder/websocket"

	"github.com/regularkevvv/agentic/realtime"
)

// dialect is what differs between providers that speak the OpenAI Realtime
// event family. Event names are shared; session shape, history content types,
// and interruption support are not.
type dialect struct {
	name             string
	url              string
	session          func(realtime.Config) map[string]any
	assistantContent string
	// truncates reports whether the provider accepts
	// conversation.item.truncate to drop audio the user never heard.
	truncates bool
}

// providerConn is the gateway's server-held WebSocket to one provider call.
// It is the provider leg of a relayed call: the bridge sees its control
// events through Recv, while output audio is handed to onAudio and never
// reaches the bridge. Both legs use G.711 μ-law at 8 kHz, so audio passes
// through without transcoding.
type providerConn struct {
	socket  *websocket.Conn
	dialect dialect
	onAudio func(itemID string, mulaw []byte)
}

func dialProvider(ctx context.Context, d dialect, key string) (*providerConn, error) {
	socket, response, err := websocket.Dial(ctx, d.url, &websocket.DialOptions{
		HTTPHeader: http.Header{"Authorization": {"Bearer " + key}},
	})
	if err != nil {
		if response != nil {
			return nil, fmt.Errorf("dial %s: %s: %w", d.name, response.Status, err)
		}
		return nil, fmt.Errorf("dial %s: %w", d.name, err)
	}
	socket.SetReadLimit(16 << 20)
	return &providerConn{socket: socket, dialect: d}, nil
}

func (c *providerConn) Close() error { return c.socket.Close(websocket.StatusNormalClosure, "") }

func (c *providerConn) write(ctx context.Context, event map[string]any) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	return c.socket.Write(ctx, websocket.MessageText, data)
}

func message(role, contentType, text string) map[string]any {
	return map[string]any{"type": "conversation.item.create", "item": map[string]any{
		"type": "message", "role": role, "content": []any{map[string]any{"type": contentType, "text": text}},
	}}
}

func (c *providerConn) Send(ctx context.Context, action realtime.Action) error {
	switch action.Kind {
	case realtime.ActionConfigure:
		return c.write(ctx, map[string]any{"type": "session.update", "session": c.dialect.session(*action.Config)})
	case realtime.ActionHistory:
		for _, turn := range action.History {
			event := message("user", "input_text", turn.Text)
			if turn.Role == realtime.RoleAssistant {
				event = message("assistant", c.dialect.assistantContent, turn.Text)
			}
			if err := c.write(ctx, event); err != nil {
				return err
			}
		}
		return nil
	case realtime.ActionUserText:
		return c.write(ctx, message("user", "input_text", action.Text))
	case realtime.ActionToolResult:
		return c.write(ctx, map[string]any{"type": "conversation.item.create", "item": map[string]any{
			"type": "function_call_output", "call_id": action.Result.CallID, "output": action.Result.Output,
		}})
	case realtime.ActionRespond:
		return c.write(ctx, map[string]any{"type": "response.create"})
	case realtime.ActionCancel:
		return c.write(ctx, map[string]any{"type": "response.cancel"})
	}
	return fmt.Errorf("unsupported action %q", action.Kind)
}

// appendAudio forwards one frame of the user's μ-law audio.
func (c *providerConn) appendAudio(ctx context.Context, mulaw []byte) error {
	return c.write(ctx, map[string]any{
		"type": "input_audio_buffer.append", "audio": base64.StdEncoding.EncodeToString(mulaw),
	})
}

// truncate tells the provider how much of an interrupted item the user
// actually heard, so its context matches what was played.
func (c *providerConn) truncate(ctx context.Context, itemID string, playedMs int) error {
	if !c.dialect.truncates || itemID == "" {
		return nil
	}
	return c.write(ctx, map[string]any{
		"type": "conversation.item.truncate", "item_id": itemID, "content_index": 0, "audio_end_ms": playedMs,
	})
}

// Recv returns the next neutral event. Output audio goes to onAudio;
// bookkeeping events the frontend does not act on are skipped.
func (c *providerConn) Recv(ctx context.Context) (realtime.Event, error) {
	for {
		_, data, err := c.socket.Read(ctx)
		if err != nil {
			return realtime.Event{}, err
		}
		var wire struct {
			Type       string                   `json:"type"`
			Delta      string                   `json:"delta"`
			ItemID     string                   `json:"item_id"`
			Transcript string                   `json:"transcript"`
			Text       string                   `json:"text"`
			CallID     string                   `json:"call_id"`
			Name       string                   `json:"name"`
			Arguments  string                   `json:"arguments"`
			Response   struct{ Status string }  `json:"response"`
			Error      struct{ Message string } `json:"error"`
		}
		if err := json.Unmarshal(data, &wire); err != nil {
			return realtime.Event{}, fmt.Errorf("decode event: %w", err)
		}
		switch wire.Type {
		case "response.output_audio.delta":
			if c.onAudio != nil {
				audio, err := base64.StdEncoding.DecodeString(wire.Delta)
				if err != nil {
					return realtime.Event{}, fmt.Errorf("decode audio: %w", err)
				}
				c.onAudio(wire.ItemID, audio)
			}
		case "session.updated":
			return realtime.Event{Kind: realtime.EventReady}, nil
		case "input_audio_buffer.speech_started":
			return realtime.Event{Kind: realtime.EventSpeechStarted}, nil
		case "input_audio_buffer.speech_stopped":
			return realtime.Event{Kind: realtime.EventSpeechStopped}, nil
		case "conversation.item.input_audio_transcription.completed":
			return realtime.Event{Kind: realtime.EventUserTranscript, Text: wire.Transcript}, nil
		case "response.output_audio_transcript.done":
			return realtime.Event{Kind: realtime.EventAssistantTranscript, Text: wire.Transcript}, nil
		case "response.output_text.done":
			return realtime.Event{Kind: realtime.EventAssistantTranscript, Text: wire.Text}, nil
		case "response.function_call_arguments.done":
			return realtime.Event{Kind: realtime.EventToolCall, Call: &realtime.ToolCall{
				ID: wire.CallID, Name: wire.Name, Arguments: json.RawMessage(wire.Arguments),
			}}, nil
		case "response.created":
			return realtime.Event{Kind: realtime.EventResponseStarted}, nil
		case "response.done":
			return realtime.Event{Kind: realtime.EventResponseDone, Text: wire.Response.Status}, nil
		case "error":
			return realtime.Event{Kind: realtime.EventError, Text: wire.Error.Message}, nil
		}
	}
}

func functionTools(tools []realtime.ToolSpec) []any {
	out := make([]any, 0, len(tools))
	for _, tool := range tools {
		out = append(out, map[string]any{
			"type": "function", "name": tool.Name, "description": tool.Description, "parameters": tool.Parameters,
		})
	}
	return out
}

var mulaw = map[string]any{"type": "audio/pcmu"}

// openAIDialect is the GA Realtime session shape: typed session, audio
// settings nested under audio.input and audio.output.
var openAIDialect = dialect{
	name:             "openai",
	url:              "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1",
	assistantContent: "output_text",
	truncates:        true,
	session: func(config realtime.Config) map[string]any {
		return map[string]any{
			"type":         "realtime",
			"instructions": config.Instructions,
			"tools":        functionTools(config.Tools),
			"audio": map[string]any{
				"input": map[string]any{
					"format":         mulaw,
					"transcription":  map[string]any{"model": "gpt-4o-mini-transcribe"},
					"turn_detection": map[string]any{"type": "semantic_vad"},
				},
				"output": map[string]any{"format": mulaw, "voice": config.Voice},
			},
		}
	},
}

// grokDialect is xAI's shape: untyped session with voice and turn detection
// at the top level, and plain "text" assistant history content. xAI does not
// document conversation.item.truncate.
var grokDialect = dialect{
	name:             "grok",
	url:              "wss://api.x.ai/v1/realtime?model=grok-voice-latest",
	assistantContent: "text",
	session: func(config realtime.Config) map[string]any {
		return map[string]any{
			"instructions":   config.Instructions,
			"voice":          config.Voice,
			"tools":          functionTools(config.Tools),
			"turn_detection": map[string]any{"type": "server_vad"},
			"audio": map[string]any{
				"input":  map[string]any{"format": mulaw},
				"output": map[string]any{"format": mulaw},
			},
		}
	},
}
