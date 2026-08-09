module github.com/regularkevvv/agentic/e2e

go 1.25.4

require (
	github.com/regularkevvv/agentic v0.6.0
	github.com/regularkevvv/agentic/harness v0.3.0
	github.com/regularkevvv/agentic/harness/codemode/gomonty v0.1.0
	github.com/regularkevvv/agentic/tui v0.1.0
	github.com/regularkevvv/gomonty v0.0.15
)

require (
	charm.land/bubbles/v2 v2.1.1 // indirect
	charm.land/bubbletea/v2 v2.0.8 // indirect
	charm.land/lipgloss/v2 v2.0.5 // indirect
	cloud.google.com/go v0.116.0 // indirect
	cloud.google.com/go/auth v0.9.3 // indirect
	cloud.google.com/go/compute/metadata v0.5.0 // indirect
	github.com/anthropics/anthropic-sdk-go v1.27.1 // indirect
	github.com/atotto/clipboard v0.1.4 // indirect
	github.com/aws/aws-sdk-go-v2 v1.43.3 // indirect
	github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream v1.7.16 // indirect
	github.com/aws/aws-sdk-go-v2/config v1.32.12 // indirect
	github.com/aws/aws-sdk-go-v2/credentials v1.19.12 // indirect
	github.com/aws/aws-sdk-go-v2/feature/ec2/imds v1.18.20 // indirect
	github.com/aws/aws-sdk-go-v2/internal/configsources v1.4.34 // indirect
	github.com/aws/aws-sdk-go-v2/internal/endpoints/v2 v2.7.34 // indirect
	github.com/aws/aws-sdk-go-v2/internal/ini v1.8.6 // indirect
	github.com/aws/aws-sdk-go-v2/service/bedrockruntime v1.50.3 // indirect
	github.com/aws/aws-sdk-go-v2/service/internal/accept-encoding v1.13.7 // indirect
	github.com/aws/aws-sdk-go-v2/service/internal/presigned-url v1.13.20 // indirect
	github.com/aws/aws-sdk-go-v2/service/sagemakerruntime v1.43.3 // indirect
	github.com/aws/aws-sdk-go-v2/service/signin v1.0.8 // indirect
	github.com/aws/aws-sdk-go-v2/service/sso v1.30.13 // indirect
	github.com/aws/aws-sdk-go-v2/service/ssooidc v1.35.17 // indirect
	github.com/aws/aws-sdk-go-v2/service/sts v1.41.9 // indirect
	github.com/aws/smithy-go v1.27.6 // indirect
	github.com/charmbracelet/colorprofile v0.4.3 // indirect
	github.com/charmbracelet/ultraviolet v0.0.0-20260703014108-f5a850f9c2b7 // indirect
	github.com/charmbracelet/x/ansi v0.11.7 // indirect
	github.com/charmbracelet/x/term v0.2.2 // indirect
	github.com/charmbracelet/x/termios v0.1.1 // indirect
	github.com/charmbracelet/x/windows v0.2.2 // indirect
	github.com/clipperhouse/displaywidth v0.11.0 // indirect
	github.com/clipperhouse/uax29/v2 v2.7.0 // indirect
	github.com/ebitengine/purego v0.10.0 // indirect
	github.com/gabriel-vasile/mimetype v1.4.12 // indirect
	github.com/go-playground/locales v0.14.1 // indirect
	github.com/go-playground/universal-translator v0.18.1 // indirect
	github.com/go-playground/validator/v10 v10.30.1 // indirect
	github.com/golang/groupcache v0.0.0-20210331224755-41bb18bfe9da // indirect
	github.com/google/go-cmp v0.6.0 // indirect
	github.com/google/s2a-go v0.1.8 // indirect
	github.com/googleapis/enterprise-certificate-proxy v0.3.4 // indirect
	github.com/gorilla/websocket v1.5.3 // indirect
	github.com/leodido/go-urn v1.4.0 // indirect
	github.com/lucasb-eyer/go-colorful v1.4.0 // indirect
	github.com/mattn/go-runewidth v0.0.24 // indirect
	github.com/muesli/cancelreader v0.2.2 // indirect
	github.com/openai/openai-go v1.12.0 // indirect
	github.com/pelletier/go-toml/v2 v2.4.3 // indirect
	github.com/regularkevvv/agentic/harness/sessionloop v0.1.0 // indirect
	github.com/rivo/uniseg v0.4.7 // indirect
	github.com/swaggest/jsonschema-go v0.3.79 // indirect
	github.com/swaggest/refl v1.4.0 // indirect
	github.com/tidwall/gjson v1.18.0 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	github.com/vmihailenco/msgpack/v5 v5.4.1 // indirect
	github.com/vmihailenco/tagparser/v2 v2.0.0 // indirect
	github.com/xo/terminfo v0.0.0-20220910002029-abceb7e1c41e // indirect
	go.opencensus.io v0.24.0 // indirect
	golang.org/x/crypto v0.46.0 // indirect
	golang.org/x/net v0.47.0 // indirect
	golang.org/x/sync v0.21.0 // indirect
	golang.org/x/sys v0.46.0 // indirect
	golang.org/x/text v0.32.0 // indirect
	google.golang.org/genai v1.51.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20240903143218-8af14fe29dc1 // indirect
	google.golang.org/grpc v1.66.2 // indirect
	google.golang.org/protobuf v1.34.2 // indirect
)

// These replaces are the point of this module. The e2e programs prove the
// current checkout across the coordinated library modules. The release modules
// themselves deliberately keep replace directives out of their own go.mod
// files; e2e is repository acceptance code and must exercise this checkout.
replace github.com/regularkevvv/agentic => ..

replace github.com/regularkevvv/agentic/harness => ../harness

replace github.com/regularkevvv/agentic/harness/codemode/gomonty => ../harness/codemode/gomonty

replace github.com/regularkevvv/agentic/tui => ../tui
