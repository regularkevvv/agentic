package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"path"

	"github.com/regularkevvv/agentic/realtime"
)

// openAISignaler implements realtime.Signaler with OpenAI's unified WebRTC
// interface. The client's SDP offer is posted by the server, so the client
// never holds a credential, and the server attaches a sideband WebSocket to
// the same call. Audio flows client <-> OpenAI; the server only controls.
type openAISignaler struct {
	key   string
	model string
}

func (s openAISignaler) Accept(ctx context.Context, offer realtime.Offer) (realtime.Answer, realtime.Conn, error) {
	if offer.ContentType != "application/sdp" {
		return realtime.Answer{}, nil, fmt.Errorf("unsupported offer type %q", offer.ContentType)
	}
	session, err := json.Marshal(map[string]any{"type": "realtime", "model": s.model})
	if err != nil {
		return realtime.Answer{}, nil, err
	}
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	if err := form.WriteField("sdp", string(offer.Body)); err != nil {
		return realtime.Answer{}, nil, err
	}
	if err := form.WriteField("session", string(session)); err != nil {
		return realtime.Answer{}, nil, err
	}
	if err := form.Close(); err != nil {
		return realtime.Answer{}, nil, err
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/realtime/calls", &body)
	if err != nil {
		return realtime.Answer{}, nil, err
	}
	request.Header.Set("Authorization", "Bearer "+s.key)
	request.Header.Set("Content-Type", form.FormDataContentType())
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return realtime.Answer{}, nil, err
	}
	defer func() { _ = response.Body.Close() }()
	answer, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return realtime.Answer{}, nil, err
	}
	if response.StatusCode/100 != 2 {
		return realtime.Answer{}, nil, fmt.Errorf("create call: %s: %s", response.Status, answer)
	}
	location := response.Header.Get("Location")
	if location == "" {
		return realtime.Answer{}, nil, fmt.Errorf("create call: no Location header")
	}

	sideband, err := dial(ctx, "wss://api.openai.com/v1/realtime?call_id="+path.Base(location), s.key, openAIDialect)
	if err != nil {
		return realtime.Answer{}, nil, fmt.Errorf("attach sideband: %w", err)
	}
	return realtime.Answer{ContentType: "application/sdp", Body: answer}, sideband, nil
}
