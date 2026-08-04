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
	for _, want := range []string{"ASSISTANT", "answer", "thinking\n  secret plan", "thinking [redacted]", "• Using tools", "├─ Read", "safe", "└─ Tool", "more |", "EVENT"} {
		if !strings.Contains(visible, want) {
			t.Fatalf("visible transcript lacks %q: %q", want, visible)
		}
	}
	collapsed := Entry(entry, Options{NoColor: true, Thinking: ThinkingCollapsed})
	if !strings.Contains(collapsed, "thinking collapsed (11 chars)") || strings.Contains(collapsed, "secret plan") {
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

func TestTranscriptFoldsToolLifecycleAndFormatsMarkdown(t *testing.T) {
	t.Parallel()
	entries := []uit.Entry{
		{Role: uit.RoleAssistant, Text: "The **answer** uses `Go`.", Tools: []uit.Tool{{CallID: "call", Name: "read_file", State: uit.ToolPlanned}}},
		{Role: uit.RoleTool, Tools: []uit.Tool{{CallID: "call", Name: "read_file", State: uit.ToolDone, Presentation: uit.ToolPresentation{Category: uit.ToolCategoryExplore, Title: "Read README", Detail: "README.md"}}}},
	}
	plain := Transcript(entries, "", Options{NoColor: true, ToolExpanded: true})
	for _, want := range []string{"The answer uses Go.", "✓ Explored Read README", "README.md"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("transcript lacks %q: %q", want, plain)
		}
	}
	if strings.Count(plain, "Read README") != 1 || strings.Contains(plain, "planned") || strings.Contains(plain, "**") {
		t.Fatalf("tool lifecycle was not folded: %q", plain)
	}
	colored := Transcript(entries, "", Options{ToolExpanded: true})
	if !strings.Contains(colored, "\x1b[") {
		t.Fatalf("formatted transcript has no ANSI styling: %q", colored)
	}
	markdownValue := markdown("# Heading\n- **bold**\n  * nested\n> quote\n```go\nfmt.Println()\n```\n```\nplain\n```\nunmatched ** marker\nunmatched ` code", true)
	for _, want := range []string{"Heading", "• bold", "  • nested", "│ quote", "│ go", "│ fmt.Println()", "│ plain", "unmatched ** marker", "unmatched ` code"} {
		if !strings.Contains(markdownValue, want) {
			t.Fatalf("markdown lacks %q: %q", want, markdownValue)
		}
	}
	if coloredMarkdown := markdown("## **Heading**\n> `quote`\n```go\ncode\n```", false); !strings.Contains(coloredMarkdown, "\x1b[") {
		t.Fatalf("colored markdown has no styling: %q", coloredMarkdown)
	}
}

func TestTranscriptGroupsAdjacentToolsAndCarriesSafeCommandPresentation(t *testing.T) {
	t.Parallel()
	entries := []uit.Entry{
		{Role: uit.RoleAssistant, Text: "I will check."},
		{Role: uit.RoleAssistant, Tools: []uit.Tool{{
			CallID: "one", Name: "run_command", State: uit.ToolPlanned, Summary: "rg -n TODO README.md",
			Presentation: uit.ToolPresentation{Category: uit.ToolCategoryExecute, Title: "rg -n TODO README.md"},
		}}},
		{Role: uit.RoleTool, Tools: []uit.Tool{{CallID: "one", Name: "run_command", State: uit.ToolDone, Presentation: uit.ToolPresentation{Category: uit.ToolCategoryExecute, Title: "Run command"}}}},
		{Role: uit.RoleAssistant, Tools: []uit.Tool{{
			CallID: "two", Name: "run_command", State: uit.ToolDone,
			Presentation: uit.ToolPresentation{Category: uit.ToolCategoryExecute, Title: "go test ./..."},
		}}},
		{Role: uit.RoleTool, Tools: []uit.Tool{{
			CallID: "three", Name: "read_file", State: uit.ToolDone,
			Presentation: uit.ToolPresentation{Category: uit.ToolCategoryExplore, Title: "Read README.md"},
		}}},
		{Role: uit.RoleAssistant, Text: "Done."},
	}
	plain := Transcript(entries, "", Options{NoColor: true})
	for _, want := range []string{"I will check.", "✓ Ran", "├─ rg -n TODO README.md", "└─ go test ./...", "✓ Explored Read README.md", "Done."} {
		if !strings.Contains(plain, want) {
			t.Fatalf("grouped transcript lacks %q: %q", want, plain)
		}
	}
	if strings.Count(plain, "rg -n TODO README.md") != 1 || strings.Contains(plain, "Run command") {
		t.Fatalf("tool lifecycle did not preserve the safe command: %q", plain)
	}
}

func TestToolPresentationStatesCategoriesAndFallbacks(t *testing.T) {
	t.Parallel()
	states := []uit.ToolState{uit.ToolPreview, uit.ToolPlanned, uit.ToolRunning, uit.ToolDone, uit.ToolError}
	for _, state := range states {
		tool := uit.Tool{State: state, Summary: "safe detail"}
		plain := Tool(tool, Options{NoColor: true, ToolExpanded: true})
		if !strings.Contains(plain, "Tool") || !strings.Contains(plain, "safe detail") {
			t.Fatalf("plain %s tool = %q", state, plain)
		}
		if colored := Tool(tool, Options{ToolExpanded: true}); !strings.Contains(colored, "\x1b[") {
			t.Fatalf("colored %s tool = %q", state, colored)
		}
	}
	tools := []uit.Tool{
		{Name: "read_file", State: uit.ToolDone},
		{Name: "run_command", State: uit.ToolRunning},
		{Name: "write_file", State: uit.ToolError},
		{Name: "custom_name", State: uit.ToolPlanned},
		{Name: "custom_category", State: uit.ToolDone, Presentation: uit.ToolPresentation{Category: uit.ToolCategory("database")}},
	}
	plain := Tools(tools, Options{NoColor: true})
	for _, want := range []string{"✓ Explored", "• Running", "✗ Change failed", "○ Using tools", "✓ Used tools", "Read file", "Run command", "Write file", "Custom name"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("grouped tools lack %q: %q", want, plain)
		}
	}
	if got := aggregateToolState([]uit.Tool{{State: uit.ToolDone}, {State: uit.ToolPlanned}}); got != uit.ToolPlanned {
		t.Fatalf("planned aggregate = %s", got)
	}
	if got := aggregateToolState([]uit.Tool{{State: uit.ToolRunning}, {State: uit.ToolError}}); got != uit.ToolError {
		t.Fatalf("error aggregate = %s", got)
	}
	for name, want := range map[string]uit.ToolCategory{
		"list_files": uit.ToolCategoryExplore, "stat_file": uit.ToolCategoryExplore,
		"read_artifact": uit.ToolCategoryExplore, "make_directory": uit.ToolCategoryChange,
		"remove_path": uit.ToolCategoryChange,
	} {
		if got := inferCategory(name); got != want {
			t.Fatalf("%s category = %s", name, got)
		}
	}
}

func TestApprovalStatusFooterAndBanner(t *testing.T) {
	t.Parallel()
	if Approval(nil, 0, nil, true) != "" {
		t.Fatal("nil approval rendered")
	}
	if got := ApprovalResolving(true, 12); got != "Applying ..." {
		t.Fatalf("resolving approval = %q", got)
	}
	if got := ApprovalResolving(false); !strings.Contains(got, "\x1b[") {
		t.Fatalf("colored resolving approval = %q", got)
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
	if !strings.Contains(Footer(uit.StateIdle, true), "enter send") || !strings.Contains(Footer(uit.StateIdle, true), "pgup/pgdn scroll") || !strings.Contains(Footer(uit.StateSuspended, false), "permission review") {
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
