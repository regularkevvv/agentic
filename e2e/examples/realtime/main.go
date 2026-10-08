// Example: a WebRTC voice gateway in front of a durable Harness session.
//
// The browser connects only to this server: an authenticated HTTPS POST of
// its SDP offer, then a μ-law audio track each way and a control data
// channel. The gateway relays audio to a voice provider (OpenAI Realtime or
// xAI Grok Voice) over a server-held WebSocket, and realtime.Run bridges the
// voice model's single delegate tool to the user's Harness session, which
// runs its own model and tools.
//
//	go run ./e2e/examples/realtime -provider grok   -client headless
//	go run ./e2e/examples/realtime -provider openai -client chrome
//	go run ./e2e/examples/realtime -provider openai -client none   # open the printed URL
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

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

const demoToken = "demo-token"

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
	provider := flag.String("provider", "grok", "voice provider behind the gateway: openai or grok")
	client := flag.String("client", "headless", "headless, chrome, or none to serve the page for a person")
	addr := flag.String("addr", "127.0.0.1:0", "gateway listen address")
	say := flag.String("say", "What's today's forecast for Lima? Do I need an umbrella?", "the spoken question")
	chrome := flag.String("chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "Chrome binary for -client chrome")
	flag.Parse()
	if err := envutil.LoadDotEnv(); err != nil {
		log.Fatal(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if *client != "none" {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, 2*time.Minute)
		defer cancel()
	}
	if err := run(ctx, *provider, *client, *addr, *say, *chrome); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, provider, client, addr, say, chrome string) error {
	var (
		d     dialect
		key   string
		voice string
	)
	switch provider {
	case "openai":
		d, key, voice = openAIDialect, os.Getenv("OPENAI_API_KEY"), "marin"
	case "grok":
		d, key, voice = grokDialect, os.Getenv("GROK_API_KEY"), "eve"
	default:
		return fmt.Errorf("unknown provider %q", provider)
	}
	host, err := newHost()
	if err != nil {
		return err
	}
	life, cancel := context.WithCancel(ctx)
	defer cancel()
	gw, err := newGateway(life, d, key)
	if err != nil {
		return err
	}
	observed := make(chan realtime.Event, 256)
	srv := &server{
		gateway:  gw,
		sessions: newSessions(host),
		tokens:   map[string]string{demoToken: "demo-user"},
		options: realtime.Options{Voice: voice, Observe: func(event realtime.Event) {
			select {
			case observed <- event:
			default:
			}
		}},
	}
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	httpServer := &http.Server{Handler: srv.routes(), ReadHeaderTimeout: 10 * time.Second}
	go func() { _ = httpServer.Serve(listener) }()
	defer func() {
		cancel()
		_ = httpServer.Shutdown(context.Background())
		srv.calls.Wait()
	}()
	base := "http://" + listener.Addr().String()
	fmt.Printf("gateway  %s  (provider %s, server-side only)\n", base, provider)

	switch client {
	case "none":
		fmt.Printf("open %s/ and press Start call (token %q)\n", base, demoToken)
		<-ctx.Done()
		return nil
	case "headless":
		return proveHeadless(ctx, base, say)
	case "chrome":
		return proveChrome(ctx, base, say, chrome, observed)
	}
	return fmt.Errorf("unknown client %q", client)
}

// proveHeadless drives the gateway the way the browser page does and checks
// what the browser would receive.
func proveHeadless(ctx context.Context, base, say string) error {
	if _, status, err := postOffer(ctx, base+"/call", "", "v=0"); err != nil || status != http.StatusUnauthorized {
		return fmt.Errorf("FAIL: unauthenticated offer got %d %v, want 401", status, err)
	}
	fmt.Println("auth     unauthenticated offer rejected with 401")

	speech, err := synthesize(ctx, os.Getenv("OPENAI_API_KEY"), say)
	if err != nil {
		return err
	}
	c, err := callGateway(ctx, base+"/call", demoToken)
	if err != nil {
		return err
	}
	defer c.close()

	var delegated bool
	var heard, answer string
	for answer == "" {
		select {
		case msg := <-c.messages:
			fmt.Printf("browser  <- %-16s %s\n", msg.Type, msg.Text)
			switch msg.Type {
			case "ready":
				fmt.Printf("browser  -> speaking %.1fs of audio: %q\n", float64(len(speech))/24000, say)
				c.say(mulawFrames(speech))
			case "transcript.user":
				heard = msg.Text
			case "delegating":
				delegated = true
			case "transcript.agent":
				if delegated {
					answer = msg.Text
				}
			}
		case <-ctx.Done():
			return errors.New("FAIL: timed out waiting for the spoken answer")
		}
	}
	// The gateway paces audio in real time, so playback trails the transcript.
	deadline := time.Now().Add(10 * time.Second)
	for c.packets.Load() < 100 && time.Now().Before(deadline) {
		time.Sleep(100 * time.Millisecond)
	}
	spokenMs, err := bargeIn(ctx, c)
	if err != nil {
		return err
	}

	fmt.Printf("\nprovider heard:      %q\n", heard)
	fmt.Printf("harness tool calls:  %d\n", forecastCalls.Load())
	fmt.Printf("browser audio:       %d RTP packets, %d bytes of μ-law from the gateway\n", c.packets.Load(), c.bytes.Load())
	fmt.Printf("barge-in:            answer cut after %d ms; unplayed audio dropped at the gateway\n", spokenMs)
	if forecastCalls.Load() == 0 || !strings.Contains(answer, "17") {
		return errors.New("FAIL: the spoken answer did not come from the Harness tool")
	}
	if c.packets.Load() == 0 {
		return errors.New("FAIL: no agent audio reached the browser")
	}
	fmt.Println("PASS")
	return nil
}

// bargeIn talks over the agent's answer while the gateway is still pacing it
// out, and expects the gateway to cut playback without a provider error.
func bargeIn(ctx context.Context, c *headlessClient) (int, error) {
	interjection, err := synthesize(ctx, os.Getenv("OPENAI_API_KEY"), "Wait, stop. That's enough, thanks.")
	if err != nil {
		return 0, err
	}
	fmt.Println("browser  -> talking over the answer")
	c.say(mulawFrames(interjection))
	timeout := time.After(15 * time.Second)
	for {
		select {
		case msg := <-c.messages:
			fmt.Printf("browser  <- %-16s %s\n", msg.Type, msg.Text)
			switch msg.Type {
			case "error":
				return 0, fmt.Errorf("FAIL: provider error during barge-in: %s", msg.Text)
			case "interrupted":
				// A rejected truncate would arrive as an error shortly after.
				settle := time.After(2 * time.Second)
				for {
					select {
					case late := <-c.messages:
						if late.Type == "error" {
							return 0, fmt.Errorf("FAIL: provider rejected the truncation: %s", late.Text)
						}
					case <-settle:
						return msg.SpokenMs, nil
					}
				}
			}
		case <-timeout:
			return 0, errors.New("FAIL: talking over the answer did not interrupt it")
		case <-ctx.Done():
			return 0, ctx.Err()
		}
	}
}

// proveChrome loads the real page in headless Chrome with the spoken question
// as its microphone, and checks the call from the server's side.
func proveChrome(ctx context.Context, base, say, chrome string, observed <-chan realtime.Event) error {
	speech, err := synthesize(ctx, os.Getenv("OPENAI_API_KEY"), say)
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "realtime-chrome-")
	if err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(dir) }()
	// Leading silence covers page load and call setup; trailing silence lets
	// voice activity detection see the end of the turn.
	padded := append(append(make([]int16, 24000*4), speech...), make([]int16, 24000*3)...)
	wav := filepath.Join(dir, "question.wav")
	if err := writeWAV(wav, padded, 24000); err != nil {
		return err
	}
	browser := exec.CommandContext(ctx, chrome,
		"--headless=new", "--no-first-run", "--no-default-browser-check", "--user-data-dir="+filepath.Join(dir, "profile"),
		"--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
		"--use-file-for-fake-audio-capture="+wav+"%noloop", "--autoplay-policy=no-user-gesture-required",
		"--disable-features=WebRtcHideLocalIpsWithMdns,AudioServiceSandbox,AudioServiceOutOfProcess",
		base+"/?autostart=1&token="+demoToken)
	if err := browser.Start(); err != nil {
		return fmt.Errorf("start Chrome: %w", err)
	}
	defer func() { _ = browser.Process.Kill(); _ = browser.Wait() }()
	fmt.Println("chrome   page loaded with the spoken question as its microphone")

	var delegated bool
	for {
		select {
		case event := <-observed:
			switch event.Kind {
			case realtime.EventUserTranscript:
				fmt.Printf("server   heard     %q\n", event.Text)
			case realtime.EventToolCall:
				delegated = true
				fmt.Printf("server   delegate  %s\n", event.Call.Arguments)
			case realtime.EventAssistantTranscript:
				fmt.Printf("server   spoke     %q\n", event.Text)
				if delegated {
					fmt.Printf("\nharness tool calls:  %d\n", forecastCalls.Load())
					if forecastCalls.Load() == 0 || !strings.Contains(event.Text, "17") {
						return errors.New("FAIL: the spoken answer did not come from the Harness tool")
					}
					fmt.Println("PASS")
					return nil
				}
			case realtime.EventError:
				fmt.Printf("server   error     %s\n", event.Text)
			}
		case <-ctx.Done():
			return errors.New("FAIL: timed out waiting for the browser call to be answered")
		}
	}
}

// newHost assembles an in-memory Harness with a text model and one tool,
// exposed through the neutral session protocol.
func newHost() (sessionloop.Host, error) {
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
	return harness.NewSessionLoopHost(runtime)
}
