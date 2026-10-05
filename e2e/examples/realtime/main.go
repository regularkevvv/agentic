// Example: a live voice call bridged to a durable Harness session.
//
// The voice model (OpenAI Realtime or xAI Grok Voice) holds the call and
// delegates every request to a Harness session through realtime.Run. The
// Harness runs its own text model and tools; the voice model only speaks the
// committed result. One typed user turn stands in for microphone audio so the
// proof runs headless.
//
//	go run ./e2e/examples/realtime -provider grok
//	go run ./e2e/examples/realtime -provider openai
//	go run ./e2e/examples/realtime -provider openai -transport webrtc
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/e2e/examples/internal/envutil"
	"github.com/regularkevvv/agentic/harness"
	artifactmemory "github.com/regularkevvv/agentic/harness/artifact/memory"
	"github.com/regularkevvv/agentic/harness/artifact/spill"
	jsoncodec "github.com/regularkevvv/agentic/harness/codec/json"
	envmemory "github.com/regularkevvv/agentic/harness/env/memory"
	"github.com/regularkevvv/agentic/harness/event/inproc"
	"github.com/regularkevvv/agentic/harness/runtime/system"
	"github.com/regularkevvv/agentic/harness/sessionloop"
	storememory "github.com/regularkevvv/agentic/harness/store/memory"
	"github.com/regularkevvv/agentic/provider/openai"
	"github.com/regularkevvv/agentic/realtime"
)

// ForecastInput is the Harness tool. The voice model never sees it.
type ForecastInput struct {
	_    struct{} `tool:"Look up today's forecast for a city"`
	City string   `json:"city" description:"City name"`
}

var forecastCalls atomic.Int32

func forecast(_ context.Context, input ForecastInput) (string, error) {
	forecastCalls.Add(1)
	return fmt.Sprintf("%s: 17°C, garúa drizzle until 11am, then overcast. Umbrella optional.", input.City), nil
}

func main() {
	provider := flag.String("provider", "grok", "voice provider: openai or grok")
	transport := flag.String("transport", "ws", "ws, or webrtc (openai only)")
	say := flag.String("say", "What's today's forecast for Lima? Do I need an umbrella?", "the user's turn")
	flag.Parse()
	if err := envutil.LoadDotEnv(); err != nil {
		log.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	if err := run(ctx, *provider, *transport, *say); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, provider, transport, say string) error {
	session, err := newSession(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = session.Close(context.Background()) }()

	var (
		conn   realtime.Conn
		client *webrtcClient
		voice  string
	)
	switch {
	case provider == "grok" && transport == "ws":
		voice = "eve"
		conn, err = dial(ctx, "wss://api.x.ai/v1/realtime?model=grok-voice-latest", os.Getenv("GROK_API_KEY"), grokDialect)
	case provider == "openai" && transport == "ws":
		voice = "marin"
		conn, err = dial(ctx, "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1", os.Getenv("OPENAI_API_KEY"), openAIDialect)
	case provider == "openai" && transport == "webrtc":
		voice = "marin"
		client, conn, err = newWebRTCClient(ctx, openAISignaler{key: os.Getenv("OPENAI_API_KEY"), model: "gpt-realtime-2.1"})
	default:
		return fmt.Errorf("unsupported provider/transport %s/%s", provider, transport)
	}
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close() }()
	if client != nil {
		defer client.close()
	}

	ready := make(chan struct{}, 1)
	answered := make(chan string, 1)
	delegated := false
	observe := func(event realtime.Event) {
		switch event.Kind {
		case realtime.EventReady:
			select {
			case ready <- struct{}{}:
			default:
			}
		case realtime.EventToolCall:
			delegated = true
			fmt.Printf("voice  -> %s(%s)\n", event.Call.Name, event.Call.Arguments)
		case realtime.EventAssistantTranscript:
			fmt.Printf("voice  <- %q\n", event.Text)
			if delegated {
				select {
				case answered <- event.Text:
				default:
				}
			}
		case realtime.EventError:
			fmt.Printf("voice  !! %s\n", event.Text)
		}
	}
	bridged := make(chan error, 1)
	go func() {
		bridged <- realtime.Run(ctx, conn, session, realtime.Options{Voice: voice, Observe: observe})
	}()

	select {
	case <-ready:
	case err := <-bridged:
		return fmt.Errorf("bridge ended before the session was configured: %w", err)
	}
	fmt.Printf("user   -> %q\n", say)
	if client != nil {
		// The browser's side of the call: the turn enters on the client's
		// data channel; the server sees it only through the sideband.
		err = client.say(say)
	} else {
		err = sendTurn(ctx, conn, say)
	}
	if err != nil {
		return err
	}

	var spoken string
	select {
	case spoken = <-answered:
	case err := <-bridged:
		return fmt.Errorf("bridge ended before an answer: %w", err)
	case <-ctx.Done():
		return errors.New("timed out waiting for the spoken answer")
	}

	snapshot, err := session.Snapshot(ctx)
	if err != nil {
		return err
	}
	fmt.Printf("\nharness tool calls: %d\n", forecastCalls.Load())
	fmt.Printf("session entries:    %d (state %s)\n", len(snapshot.Entries), snapshot.State)
	if client != nil {
		fmt.Printf("client audio:       %d RTP packets, %d bytes received directly from the provider\n",
			client.packets.Load(), client.bytes.Load())
	}
	if forecastCalls.Load() == 0 || !strings.Contains(spoken, "17") {
		return errors.New("FAIL: the spoken answer did not come from the Harness tool")
	}
	if client != nil && client.packets.Load() == 0 {
		return errors.New("FAIL: no audio reached the WebRTC client")
	}
	fmt.Println("PASS")
	return nil
}

func sendTurn(ctx context.Context, conn realtime.Conn, text string) error {
	if err := conn.Send(ctx, realtime.Action{Kind: realtime.ActionUserText, Text: text}); err != nil {
		return err
	}
	return conn.Send(ctx, realtime.Action{Kind: realtime.ActionRespond})
}

// newSession assembles an in-memory Harness with a text model and one tool,
// exposed through the neutral session protocol.
func newSession(ctx context.Context) (sessionloop.Session, error) {
	model, err := openai.New("gpt-4o-mini")
	if err != nil {
		return nil, err
	}
	agent := agentic.NewAgent("You are a concise local-weather assistant. Use the forecast tool.", model)
	agentic.AddTool(agent, forecast)

	environments, err := envmemory.NewFactory(envmemory.Config{Cwd: "/workspace"})
	if err != nil {
		return nil, err
	}
	processors, err := spill.NewFactory(artifactmemory.New(), spill.Config{})
	if err != nil {
		return nil, err
	}
	runtime, err := harness.NewRuntime[string](agent, harness.RuntimeConfig{
		Sessions:         storememory.New(),
		Codec:            jsoncodec.New(),
		Events:           inproc.NewFactory(),
		Environments:     environments,
		ResultProcessors: processors,
		Clock:            system.NewClock(),
		IDs:              system.NewIDs(),
	})
	if err != nil {
		return nil, err
	}
	host, err := harness.NewSessionLoopHost(runtime)
	if err != nil {
		return nil, err
	}
	return host.NewSession(ctx, sessionloop.SessionOptions{})
}

// webrtcClient plays the browser: it owns the media connection and the
// events data channel, and holds no credential.
type webrtcClient struct {
	pc      *webrtc.PeerConnection
	events  *webrtc.DataChannel
	open    chan struct{}
	stop    chan struct{}
	packets atomic.Int64
	bytes   atomic.Int64
}

func newWebRTCClient(ctx context.Context, signaler realtime.Signaler) (*webrtcClient, realtime.Conn, error) {
	pc, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		return nil, nil, err
	}
	c := &webrtcClient{pc: pc, open: make(chan struct{}), stop: make(chan struct{})}
	microphone, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2}, "audio", "client")
	if err != nil {
		return nil, nil, err
	}
	if _, err := pc.AddTrack(microphone); err != nil {
		return nil, nil, err
	}
	pc.OnTrack(func(track *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		for {
			packet, _, err := track.ReadRTP()
			if err != nil {
				return
			}
			c.packets.Add(1)
			c.bytes.Add(int64(len(packet.Payload)))
		}
	})
	if c.events, err = pc.CreateDataChannel("oai-events", nil); err != nil {
		return nil, nil, err
	}
	c.events.OnOpen(func() { close(c.open) })

	offer, err := pc.CreateOffer(nil)
	if err != nil {
		return nil, nil, err
	}
	gathered := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(offer); err != nil {
		return nil, nil, err
	}
	<-gathered

	answer, conn, err := signaler.Accept(ctx, realtime.Offer{ContentType: "application/sdp", Body: []byte(pc.LocalDescription().SDP)})
	if err != nil {
		return nil, nil, err
	}
	if err := pc.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeAnswer, SDP: string(answer.Body)}); err != nil {
		_ = conn.Close()
		return nil, nil, err
	}
	go c.silence(microphone)
	return c, conn, nil
}

// silence keeps the microphone track live with 20ms Opus silence frames.
func (c *webrtcClient) silence(track *webrtc.TrackLocalStaticSample) {
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-c.stop:
			return
		case <-ticker.C:
			_ = track.WriteSample(media.Sample{Data: []byte{0xf8, 0xff, 0xfe}, Duration: 20 * time.Millisecond})
		}
	}
}

func (c *webrtcClient) say(text string) error {
	select {
	case <-c.open:
	case <-time.After(15 * time.Second):
		return errors.New("events data channel never opened")
	}
	item := fmt.Sprintf(`{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":%q}]}}`, text)
	if err := c.events.SendText(item); err != nil {
		return err
	}
	return c.events.SendText(`{"type":"response.create"}`)
}

func (c *webrtcClient) close() {
	close(c.stop)
	_ = c.pc.Close()
}
