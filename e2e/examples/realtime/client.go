package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
)

// headlessClient plays the browser exactly as index.html does: it posts an
// authenticated SDP offer to the gateway, sends microphone audio on a track,
// receives the agent's audio on a track, and reads the gateway's control
// messages. It never learns which provider is behind the gateway.
type headlessClient struct {
	pc         *webrtc.PeerConnection
	mic        *webrtc.TrackLocalStaticSample
	control    *webrtc.DataChannel
	messages   chan clientMessage
	utterances chan [][]byte
	open       chan struct{}
	packets    atomic.Int64
	bytes      atomic.Int64
}

func callGateway(ctx context.Context, url, token string) (*headlessClient, error) {
	engine := &webrtc.MediaEngine{}
	if err := engine.RegisterDefaultCodecs(); err != nil {
		return nil, err
	}
	pc, err := webrtc.NewAPI(webrtc.WithMediaEngine(engine)).NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		return nil, err
	}
	c := &headlessClient{
		pc: pc, messages: make(chan clientMessage, 64), utterances: make(chan [][]byte, 4), open: make(chan struct{}),
	}
	if c.mic, err = webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypePCMU, ClockRate: 8000, Channels: 1}, "mic", "client"); err != nil {
		return nil, err
	}
	if _, err := pc.AddTrack(c.mic); err != nil {
		return nil, err
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
	if c.control, err = pc.CreateDataChannel("control", nil); err != nil {
		return nil, err
	}
	c.control.OnOpen(func() { close(c.open) })
	c.control.OnMessage(func(raw webrtc.DataChannelMessage) {
		var msg clientMessage
		if json.Unmarshal(raw.Data, &msg) == nil {
			c.messages <- msg
		}
	})

	offer, err := pc.CreateOffer(nil)
	if err != nil {
		return nil, err
	}
	gathered := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(offer); err != nil {
		return nil, err
	}
	<-gathered
	answer, status, err := postOffer(ctx, url, token, pc.LocalDescription().SDP)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		_ = pc.Close()
		return nil, fmt.Errorf("gateway rejected the call: %d %s", status, answer)
	}
	if err := pc.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeAnswer, SDP: answer}); err != nil {
		return nil, err
	}
	go c.microphone(ctx)
	return c, nil
}

func postOffer(ctx context.Context, url, token, sdp string) (string, int, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBufferString(sdp))
	if err != nil {
		return "", 0, err
	}
	request.Header.Set("Content-Type", "application/sdp")
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return "", 0, err
	}
	defer func() { _ = response.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(response.Body, 64<<10))
	return string(body), response.StatusCode, err
}

// microphone streams queued utterances in real time and silence between
// them, like an open microphone, so voice activity detection sees each turn
// start and end.
func (c *headlessClient) microphone(ctx context.Context) {
	silence := bytes.Repeat([]byte{0xFF}, frameBytes)
	ticker := time.NewTicker(frame)
	defer ticker.Stop()
	var current [][]byte
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
		if len(current) == 0 {
			select {
			case current = <-c.utterances:
			default:
			}
		}
		data := silence
		if len(current) > 0 {
			data, current = current[0], current[1:]
		}
		if err := c.mic.WriteSample(media.Sample{Data: data, Duration: frame}); err != nil {
			return
		}
	}
}

func (c *headlessClient) say(frames [][]byte) { c.utterances <- frames }

func (c *headlessClient) close() { _ = c.pc.Close() }
