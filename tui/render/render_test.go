package render

import (
	"strings"
	"testing"

	uit "github.com/regularkevvv/agentic/tui"
)

func TestTranscriptModesAndTools(t *testing.T) {
	t.Parallel()
	if got := Transcript(nil, "", Options{NoColor: true}); !strings.Contains(got, "No messages") {
		t.Fatalf("empty transcript = %q", got)
	}
	entry := uit.Entry{
		Role: uit.RoleAssistant, Text: "answer",
		Thinking: []uit.Thinking{{Text: "secret plan"}, {Text: "hidden", Redacted: true}},
		Tools:    []uit.Tool{{Name: "read", State: uit.ToolDone, Summary: "safe"}, {State: uit.ToolRunning}},
	}
	visible := Transcript([]uit.Entry{entry, {Text: "event"}}, "more", Options{NoColor: true, Thinking: ThinkingVisible, ToolExpanded: true})
	for _, want := range []string{"ASSISTANT", "answer", "thinking: secret plan", "thinking [redacted]", "[done] read: safe", "[running] tool", "more |", "EVENT"} {
		if !strings.Contains(visible, want) {
			t.Fatalf("visible transcript lacks %q: %q", want, visible)
		}
	}
	collapsed := Entry(entry, Options{NoColor: true, Thinking: ThinkingCollapsed})
	if !strings.Contains(collapsed, "thinking (11 chars)") || strings.Contains(collapsed, "secret plan") {
		t.Fatalf("collapsed entry = %q", collapsed)
	}
	hidden := Entry(entry, Options{NoColor: true, Thinking: ThinkingHidden})
	if strings.Contains(hidden, "thinking") {
		t.Fatalf("hidden entry = %q", hidden)
	}
	if colored := Entry(entry, Options{Thinking: ThinkingVisible}); !strings.Contains(colored, "\x1b[") {
		t.Fatalf("colored entry has no ANSI: %q", colored)
	}
}

func TestApprovalStatusFooterAndBanner(t *testing.T) {
	t.Parallel()
	if Approval(nil, 0, nil, true) != "" {
		t.Fatal("nil approval rendered")
	}
	unsupported := Approval(&uit.Suspension{Kind: "custom", Description: "handoff"}, 0, nil, true)
	if !strings.Contains(unsupported, "custom") || !strings.Contains(unsupported, "handoff") {
		t.Fatalf("unsupported = %q", unsupported)
	}
	suspension := &uit.Suspension{Supported: true, Approvals: []uit.Approval{
		{CallID: "one", ToolName: "write", Capability: "filesystem", Action: "write", ResourceScheme: "file", CanonicalResource: "/repo/file.txt", ResourceDisplay: "file.txt"},
		{CallID: "two", ToolName: "shell", Capability: "shell", Action: "exec", ResourceScheme: "command", CanonicalResource: `{"name":"/bin/sh","args":["secret"]}`, ResourceDisplay: "/bin/sh"},
		{CallID: "three", ToolName: "opaque", Capability: "custom", Action: "use", ResourceScheme: "opaque", CanonicalResource: "raw-secret"},
	}}
	decisions := map[string]uit.DecisionAction{"one": uit.DecisionApprove}
	plain := Approval(suspension, 1, decisions, true)
	for _, want := range []string{"[approve]", "> [pending]", "filesystem/write", "file:file.txt", "shell/exec", "command:/bin/sh", "opaque:resource [details redacted]"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("approval lacks %q: %q", want, plain)
		}
	}
	for _, secret := range []string{"secret", "raw-secret", `{"name"`} {
		if strings.Contains(plain, secret) {
			t.Fatalf("approval leaked canonical data %q: %q", secret, plain)
		}
	}
	if colored := Approval(suspension, 0, decisions, false); !strings.Contains(colored, "\x1b[") {
		t.Fatalf("colored approval = %q", colored)
	}
	snapshot := uit.Snapshot{SessionID: "s", State: uit.StateRunning, ProfileLabel: "work", Workspace: "/repo", Execution: "local-host governance (not an OS sandbox)", Pending: []uit.QueuedInput{{}}, Usage: uit.Usage{PromptTokens: 100, CacheReadTokens: 80, TotalTokens: 120}}
	status := Status(snapshot, true, true)
	for _, want := range []string{"session s", "running/busy", "profile work", "cache 80.0%", "queued 1", "/repo", "exec local-host governance (not an OS sandbox)"} {
		if !strings.Contains(status, want) {
			t.Fatalf("status lacks %q: %q", want, status)
		}
	}
	defaultStatus := Status(uit.Snapshot{}, false, false)
	if !strings.Contains(defaultStatus, "custom") || !strings.Contains(defaultStatus, "\x1b[") {
		t.Fatalf("default status = %q", defaultStatus)
	}
	if !strings.Contains(Footer(uit.StateIdle, true), "enter send") || !strings.Contains(Footer(uit.StateSuspended, false), "permission review") {
		t.Fatal("footer variants missing")
	}
	if got := Status(snapshot, false, true, 20); len([]rune(got)) != 20 || !strings.HasSuffix(got, "...") {
		t.Fatalf("narrow status = %q", got)
	}
	if got := Footer(uit.StateIdle, true, 2); len([]rune(got)) != 2 {
		t.Fatalf("tiny footer = %q", got)
	}
	if got := Footer(uit.StateIdle, true, 20); len([]rune(got)) != 20 || !strings.HasSuffix(got, "...") {
		t.Fatalf("narrow footer = %q", got)
	}
	if Banner("", false, true) != "" || Banner("ok", false, true) != "ok" || Banner("bad", true, true) != "error: bad" {
		t.Fatal("plain banner variants incorrect")
	}
	if got := Banner("a long banner", false, true, 8); got != "a lon..." {
		t.Fatalf("narrow banner = %q", got)
	}
	if !strings.Contains(Banner("ok", false, false), "\x1b[") || !strings.Contains(Banner("bad", true, false), "\x1b[") {
		t.Fatal("colored banners missing ANSI")
	}
}

func TestUntrustedTextCannotEmitTerminalControlsAndSummariesAreBounded(t *testing.T) {
	t.Parallel()
	entry := uit.Entry{
		Role:     uit.Role("assistant\x1b]0;role\a"),
		Text:     "safe\x1b[31m\rrewind\a\nnext",
		Thinking: []uit.Thinking{{Text: "think\x1b[2J"}},
		Tools:    []uit.Tool{{Name: "tool\x1b[H", State: uit.ToolRunning, Summary: strings.Repeat("x", 600) + "\x1b[2J"}},
	}
	plain := Entry(entry, Options{NoColor: true, Thinking: ThinkingVisible, ToolExpanded: true})
	for _, forbidden := range []string{"\x1b", "\a", "\r"} {
		if strings.Contains(plain, forbidden) {
			t.Fatalf("terminal control %q survived: %q", forbidden, plain)
		}
	}
	if !strings.Contains(plain, "safe[31mrewind") || !strings.Contains(plain, "next") || !strings.Contains(plain, "...") || len([]rune(plain)) >= 800 {
		t.Fatalf("sanitized entry = %q", plain)
	}
	if got := boundedTerminalSafe("abcdef", 3); got != "abc..." {
		t.Fatalf("bounded text = %q", got)
	}
	if got := boundedTerminalSafe("abc", 0); got != "abc" {
		t.Fatalf("unbounded text = %q", got)
	}
}
