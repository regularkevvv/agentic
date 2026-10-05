package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/coder/websocket"

	"github.com/regularkevvv/agentic/realtime"
)

// dialect is what differs between providers that speak the OpenAI Realtime
// event family. Event names are shared; session shape and history content
// types are not.
type dialect struct {
	session          func(realtime.Config) map[string]any
	assistantContent string
}

// wsConn is a realtime.Conn over one provider WebSocket: either the whole
// call (media relayed through it) or a sideband attached to a WebRTC call.
type wsConn struct {
	socket  *websocket.Conn
	dialect dialect
}

func dial(ctx context.Context, url, key string, d dialect) (*wsConn, error) {
	socket, response, err := websocket.Dial(ctx, url, &websocket.DialOptions{
		HTTPHeader: http.Header{"Authorization": {"Bearer " + key}},
	})
	if err != nil {
		if response != nil {
			return nil, fmt.Errorf("dial %s: %s: %w", url, response.Status, err)
		}
		return nil, fmt.Errorf("dial %s: %w", url, err)
	}
	socket.SetReadLimit(16 << 20)
	return &wsConn{socket: socket, dialect: d}, nil
}

func (c *wsConn) Close() error { return c.socket.Close(websocket.StatusNormalClosure, "") }

func (c *wsConn) write(ctx context.Context, event map[string]any) error {
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

func (c *wsConn) Send(ctx context.Context, action realtime.Action) error {
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

// Recv returns the next neutral event, skipping media and bookkeeping events
// that the frontend does not act on.
func (c *wsConn) Recv(ctx context.Context) (realtime.Event, error) {
	for {
		_, data, err := c.socket.Read(ctx)
		if err != nil {
			return realtime.Event{}, err
		}
		var wire struct {
			Type       string                   `json:"type"`
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

// openAIDialect is the GA Realtime session shape: typed session, audio
// settings nested under audio.input and audio.output.
var openAIDialect = dialect{
	assistantContent: "output_text",
	session: func(config realtime.Config) map[string]any {
		return map[string]any{
			"type":         "realtime",
			"instructions": config.Instructions,
			"tools":        functionTools(config.Tools),
			"audio": map[string]any{
				"input": map[string]any{
					"transcription":  map[string]any{"model": "gpt-4o-mini-transcribe"},
					"turn_detection": map[string]any{"type": "semantic_vad"},
				},
				"output": map[string]any{"voice": config.Voice},
			},
		}
	},
}

// grokDialect is xAI's shape: untyped session with voice and turn detection
// at the top level, and plain "text" assistant history content.
var grokDialect = dialect{
	assistantContent: "text",
	session: func(config realtime.Config) map[string]any {
		return map[string]any{
			"instructions":   config.Instructions,
			"voice":          config.Voice,
			"tools":          functionTools(config.Tools),
			"turn_detection": map[string]any{"type": "server_vad"},
		}
	},
}
