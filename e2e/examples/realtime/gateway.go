package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"

	"github.com/regularkevvv/agentic/harness/sessionloop"
	"github.com/regularkevvv/agentic/realtime"
)

//go:embed index.html
var indexHTML []byte

const (
	frame        = 20 * time.Millisecond
	frameBytes   = 160 // 20 ms of 8 kHz μ-law
	playoutDelay = 60  // ms of audio assumed in the browser's jitter buffer
)

// gateway terminates browser WebRTC calls and relays each one to a provider
// over a server-held WebSocket. It implements realtime.Signaler for the
// client leg: the browser reaches only this server, never a provider.
type gateway struct {
	api     *webrtc.API
	dialect dialect
	key     string
	life    context.Context
}

func newGateway(life context.Context, d dialect, key string) (*gateway, error) {
	engine := &webrtc.MediaEngine{}
	if err := engine.RegisterCodec(webrtc.RTPCodecParameters{
		RTPCodecCapability: webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypePCMU, ClockRate: 8000, Channels: 1},
		PayloadType:        0,
	}, webrtc.RTPCodecTypeAudio); err != nil {
		return nil, err
	}
	var settings webrtc.SettingEngine
	settings.SetIncludeLoopbackCandidate(true)
	api := webrtc.NewAPI(webrtc.WithMediaEngine(engine), webrtc.WithSettingEngine(settings))
	return &gateway{api: api, dialect: d, key: key, life: life}, nil
}

// Accept answers the browser's SDP offer and returns the relayed call. The
// call is the realtime.Conn the bridge drives: its control events come from
// the provider, while audio moves between the two legs underneath it.
func (g *gateway) Accept(ctx context.Context, offer realtime.Offer) (realtime.Answer, realtime.Conn, error) {
	if offer.ContentType != "application/sdp" {
		return realtime.Answer{}, nil, fmt.Errorf("unsupported offer type %q", offer.ContentType)
	}
	life, cancel := context.WithCancel(g.life)
	c := &call{ctx: life, cancel: cancel, control: &controlChannel{}}
	ok := false
	defer func() {
		if !ok {
			_ = c.Close()
		}
	}()

	// The provider socket lives as long as the call, so it is dialed with
	// the call's context rather than the request's.
	provider, err := dialProvider(life, g.dialect, g.key)
	if err != nil {
		return realtime.Answer{}, nil, err
	}
	c.provider = provider
	provider.onAudio = c.speaker.enqueue

	if c.pc, err = g.api.NewPeerConnection(webrtc.Configuration{}); err != nil {
		return realtime.Answer{}, nil, err
	}
	c.speaker.track, err = webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypePCMU, ClockRate: 8000, Channels: 1}, "agent", "gateway")
	if err != nil {
		return realtime.Answer{}, nil, err
	}
	if _, err := c.pc.AddTrack(c.speaker.track); err != nil {
		return realtime.Answer{}, nil, err
	}
	c.pc.OnTrack(func(track *webrtc.TrackRemote, _ *webrtc.RTPReceiver) { c.listen(track) })
	c.pc.OnDataChannel(func(channel *webrtc.DataChannel) {
		if channel.Label() == "control" {
			c.control.attach(channel, c.fromClient)
		}
	})
	c.pc.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		log.Printf("call     peer connection %s", state)
		if state == webrtc.PeerConnectionStateFailed || state == webrtc.PeerConnectionStateClosed ||
			state == webrtc.PeerConnectionStateDisconnected {
			_ = c.Close()
		}
	})

	if err := c.pc.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeOffer, SDP: string(offer.Body)}); err != nil {
		return realtime.Answer{}, nil, err
	}
	answer, err := c.pc.CreateAnswer(nil)
	if err != nil {
		return realtime.Answer{}, nil, err
	}
	gathered := webrtc.GatheringCompletePromise(c.pc)
	if err := c.pc.SetLocalDescription(answer); err != nil {
		return realtime.Answer{}, nil, err
	}
	select {
	case <-gathered:
	case <-ctx.Done():
		return realtime.Answer{}, nil, ctx.Err()
	}
	go c.speaker.pace(life)
	ok = true
	return realtime.Answer{ContentType: "application/sdp", Body: []byte(c.pc.LocalDescription().SDP)}, c, nil
}

// call is one relayed browser call. It satisfies realtime.Conn by delegating
// control to the provider leg and mirroring presentation events to the
// browser's control channel.
type call struct {
	ctx      context.Context
	cancel   context.CancelFunc
	provider *providerConn
	pc       *webrtc.PeerConnection
	speaker  speaker
	control  *controlChannel
	ready    atomic.Bool
	closed   sync.Once
}

func (c *call) Send(ctx context.Context, action realtime.Action) error {
	return c.provider.Send(ctx, action)
}

func (c *call) Recv(ctx context.Context) (realtime.Event, error) {
	event, err := c.provider.Recv(ctx)
	if err != nil {
		return event, err
	}
	switch event.Kind {
	case realtime.EventReady:
		c.ready.Store(true)
		c.control.send(clientMessage{Type: "ready"})
	case realtime.EventSpeechStarted:
		c.interrupt()
		c.control.send(clientMessage{Type: "user.speaking"})
	case realtime.EventUserTranscript:
		c.control.send(clientMessage{Type: "transcript.user", Text: event.Text})
	case realtime.EventAssistantTranscript:
		c.control.send(clientMessage{Type: "transcript.agent", Text: event.Text})
	case realtime.EventToolCall:
		var args struct{ Request string }
		_ = json.Unmarshal(event.Call.Arguments, &args)
		c.control.send(clientMessage{Type: "delegating", Text: args.Request})
	case realtime.EventError:
		c.control.send(clientMessage{Type: "error", Text: event.Text})
	}
	return event, nil
}

func (c *call) Close() error {
	c.closed.Do(func() {
		c.cancel()
		if c.pc != nil {
			_ = c.pc.Close()
		}
		if c.provider != nil {
			_ = c.provider.Close()
		}
	})
	return nil
}

// listen forwards the browser's microphone to the provider once the provider
// session is configured for μ-law; earlier frames would be misread.
func (c *call) listen(track *webrtc.TrackRemote) {
	for {
		packet, _, err := track.ReadRTP()
		if err != nil {
			return
		}
		if c.ready.Load() && len(packet.Payload) > 0 {
			if err := c.provider.appendAudio(c.ctx, packet.Payload); err != nil {
				return
			}
		}
	}
}

// interrupt handles barge-in: unplayed agent audio is dropped here, and the
// provider is told how much the user actually heard.
func (c *call) interrupt() {
	itemID, playedMs, interrupted := c.speaker.flush()
	if !interrupted {
		return
	}
	if err := c.provider.truncate(c.ctx, itemID, playedMs); err != nil {
		log.Printf("truncate: %v", err)
	}
	c.control.send(clientMessage{Type: "interrupted", SpokenMs: playedMs})
}

// fromClient accepts the closed set of browser messages. Typed text always
// enters as a user turn; the browser cannot configure the voice model.
func (c *call) fromClient(msg clientMessage) {
	switch msg.Type {
	case "user.text":
		if strings.TrimSpace(msg.Text) == "" {
			return
		}
		if err := c.provider.Send(c.ctx, realtime.Action{Kind: realtime.ActionUserText, Text: msg.Text}); err == nil {
			_ = c.provider.Send(c.ctx, realtime.Action{Kind: realtime.ActionRespond})
		}
	case "hangup":
		_ = c.Close()
	}
}

// speaker paces provider audio to the browser one 20 ms frame per tick, so
// unplayed audio stays in the gateway where barge-in can drop it.
type speaker struct {
	track      *webrtc.TrackLocalStaticSample
	mu         sync.Mutex
	queue      []byte
	itemID     string
	sentMs     int
	receivedMs int
}

func (s *speaker) enqueue(itemID string, mulaw []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if itemID != s.itemID {
		s.itemID, s.sentMs, s.receivedMs = itemID, 0, 0
	}
	s.queue = append(s.queue, mulaw...)
	s.receivedMs += len(mulaw) / 8
}

func (s *speaker) pace(ctx context.Context) {
	ticker := time.NewTicker(frame)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
		s.mu.Lock()
		n := min(frameBytes, len(s.queue))
		chunk := append([]byte(nil), s.queue[:n]...)
		s.queue = s.queue[n:]
		if n > 0 {
			s.sentMs += 20
		}
		s.mu.Unlock()
		if n == 0 {
			continue
		}
		for len(chunk) < frameBytes {
			chunk = append(chunk, 0xFF) // μ-law silence
		}
		_ = s.track.WriteSample(media.Sample{Data: chunk, Duration: frame})
	}
}

// flush drops queued audio and reports how much of the current item the
// user heard. It reports no interruption when nothing was pending.
func (s *speaker) flush() (itemID string, playedMs int, interrupted bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.queue) == 0 {
		return "", 0, false
	}
	s.queue = nil
	playedMs = max(0, min(s.sentMs-playoutDelay, s.receivedMs))
	return s.itemID, playedMs, true
}

// clientMessage is the gateway's closed browser protocol. Provider events
// never pass through verbatim.
type clientMessage struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	SpokenMs int    `json:"spoken_ms,omitempty"`
}

// controlChannel buffers outbound messages until the browser's data channel
// opens.
type controlChannel struct {
	mu      sync.Mutex
	channel *webrtc.DataChannel
	open    bool
	pending []clientMessage
}

func (c *controlChannel) attach(channel *webrtc.DataChannel, receive func(clientMessage)) {
	c.mu.Lock()
	c.channel = channel
	c.mu.Unlock()
	channel.OnOpen(func() {
		c.mu.Lock()
		c.open = true
		pending := c.pending
		c.pending = nil
		c.mu.Unlock()
		for _, msg := range pending {
			c.send(msg)
		}
	})
	channel.OnMessage(func(raw webrtc.DataChannelMessage) {
		var msg clientMessage
		if json.Unmarshal(raw.Data, &msg) == nil {
			receive(msg)
		}
	})
}

func (c *controlChannel) send(msg clientMessage) {
	c.mu.Lock()
	if !c.open {
		c.pending = append(c.pending, msg)
		c.mu.Unlock()
		return
	}
	channel := c.channel
	c.mu.Unlock()
	data, _ := json.Marshal(msg)
	_ = channel.SendText(string(data))
}

// server is the application around the gateway: it authenticates the
// signaling request, picks the caller's durable session, and bridges the call
// to it. Authentication happens here, before any media is negotiated.
type server struct {
	gateway  *gateway
	sessions *sessions
	tokens   map[string]string // bearer token -> user
	options  realtime.Options
	calls    sync.WaitGroup
}

func (s *server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(indexHTML)
	})
	mux.HandleFunc("POST /call", s.handleCall)
	return mux
}

func (s *server) handleCall(w http.ResponseWriter, r *http.Request) {
	user, ok := s.tokens[strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")]
	if !ok {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	if r.Header.Get("Content-Type") != "application/sdp" {
		http.Error(w, "expected application/sdp", http.StatusUnsupportedMediaType)
		return
	}
	sdp, err := io.ReadAll(io.LimitReader(r.Body, 64<<10))
	if err != nil {
		http.Error(w, "read offer", http.StatusBadRequest)
		return
	}
	session, err := s.sessions.open(r.Context(), user)
	if errors.Is(err, errCallActive) {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	if err != nil {
		http.Error(w, "open session", http.StatusInternalServerError)
		return
	}
	answer, conn, err := s.gateway.Accept(r.Context(), realtime.Offer{ContentType: "application/sdp", Body: sdp})
	if err != nil {
		s.sessions.release(user, session)
		log.Printf("accept call for %s: %v", user, err)
		http.Error(w, "could not establish call", http.StatusBadGateway)
		return
	}
	log.Printf("call     accepted for %s", user)
	s.calls.Go(func() {
		defer s.sessions.release(user, session)
		defer func() { _ = conn.Close() }()
		if err := realtime.Run(s.gateway.life, conn, session, s.options); err != nil && s.gateway.life.Err() == nil {
			log.Printf("call for %s ended: %v", user, err)
		}
	})
	w.Header().Set("Content-Type", answer.ContentType)
	_, _ = w.Write(answer.Body)
}

var errCallActive = errors.New("a call is already active for this user")

// sessions maps each user to one durable session and admits one call per
// session at a time.
type sessions struct {
	host   sessionloop.Host
	mu     sync.Mutex
	ids    map[string]sessionloop.SessionID
	active map[string]bool
}

func newSessions(host sessionloop.Host) *sessions {
	return &sessions{host: host, ids: map[string]sessionloop.SessionID{}, active: map[string]bool{}}
}

func (s *sessions) open(ctx context.Context, user string) (sessionloop.Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.active[user] {
		return nil, errCallActive
	}
	var (
		session sessionloop.Session
		err     error
	)
	if id, ok := s.ids[user]; ok {
		session, err = s.host.OpenSession(ctx, id)
	} else {
		session, err = s.host.NewSession(ctx, sessionloop.SessionOptions{Meta: map[string]string{"user": user}})
	}
	if err != nil {
		return nil, err
	}
	s.ids[user] = session.ID()
	s.active[user] = true
	return session, nil
}

func (s *sessions) release(user string, session sessionloop.Session) {
	_ = session.Close(context.Background())
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.active, user)
}
