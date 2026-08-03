// Package render contains deterministic, terminal-independent view assembly.
package render

import (
	"fmt"
	"strings"
	"unicode"

	"charm.land/lipgloss/v2"

	uit "github.com/regularkevvv/agentic/tui"
)

type ThinkingMode string

const (
	ThinkingVisible   ThinkingMode = "visible"
	ThinkingCollapsed ThinkingMode = "collapsed"
	ThinkingHidden    ThinkingMode = "hidden"
)

type Options struct {
	Width        int
	NoColor      bool
	Thinking     ThinkingMode
	ToolExpanded bool
}

func Transcript(entries []uit.Entry, preview string, options Options) string {
	parts := make([]string, 0, len(entries)+1)
	for _, entry := range entries {
		parts = append(parts, Entry(entry, options))
	}
	if preview != "" {
		cursor := " ▌"
		if options.NoColor {
			cursor = " |"
		}
		parts = append(parts, role("assistant", options.NoColor)+"\n"+terminalSafe(preview)+cursor)
	}
	if len(parts) == 0 {
		return "No messages yet. Type a task below."
	}
	return strings.Join(parts, "\n\n")
}

func Entry(entry uit.Entry, options Options) string {
	label := string(entry.Role)
	if label == "" {
		label = "event"
	}
	lines := []string{role(label, options.NoColor)}
	if entry.Text != "" {
		lines = append(lines, terminalSafe(entry.Text))
	}
	if options.Thinking != ThinkingHidden {
		for _, thinking := range entry.Thinking {
			if options.Thinking == ThinkingCollapsed {
				lines = append(lines, fmt.Sprintf("thinking (%d chars)", len([]rune(thinking.Text))))
			} else if thinking.Redacted {
				lines = append(lines, "thinking [redacted]")
			} else {
				lines = append(lines, "thinking: "+terminalSafe(thinking.Text))
			}
		}
	}
	for _, tool := range entry.Tools {
		lines = append(lines, Tool(tool, options))
	}
	return strings.Join(lines, "\n")
}

func Tool(tool uit.Tool, options Options) string {
	name := boundedTerminalSafe(tool.Name, 128)
	if name == "" {
		name = "tool"
	}
	result := fmt.Sprintf("[%s] %s", terminalSafe(string(tool.State)), name)
	if options.ToolExpanded && tool.Summary != "" {
		result += ": " + boundedTerminalSafe(tool.Summary, 512)
	}
	return result
}

func Approval(value *uit.Suspension, selected int, decisions map[string]uit.DecisionAction, noColor bool, widths ...int) string {
	if value == nil {
		return ""
	}
	if !value.Supported {
		return "Suspended: " + terminalSafe(value.Kind) + "\n" + terminalSafe(value.Description) + "\nEsc quits safely."
	}
	lines := []string{"Permission required", "Review every requested operation:"}
	for index, approval := range value.Approvals {
		cursor := "  "
		if index == selected {
			cursor = "> "
		}
		decision := decisions[approval.CallID]
		mark := "pending"
		if decision != "" {
			mark = string(decision)
		}
		capability := boundedTerminalSafe(approval.Capability, 128)
		if capability == "" {
			capability = "unknown"
		}
		action := boundedTerminalSafe(approval.Action, 128)
		if action == "" {
			action = "unknown"
		}
		// CanonicalResource is the policy identity and can contain normalized
		// command arguments. The default terminal renders only the capability-
		// owned safe display label; an empty label remains opaque rather than
		// falling back to raw canonical data.
		resource := boundedTerminalSafe(approval.ResourceDisplay, 256)
		scheme := boundedTerminalSafe(approval.ResourceScheme, 64)
		if resource == "" {
			resource = "resource [details redacted]"
		}
		if scheme != "" {
			resource = scheme + ":" + resource
		}
		lines = append(lines, fmt.Sprintf("%s[%s] %s - %s/%s %s", cursor, mark, boundedTerminalSafe(approval.ToolName, 128), capability, action, resource))
	}
	lines = append(lines, "a approve  d deny  up/down choose  enter resume  esc handoff")
	width := firstWidth(widths)
	for index := range lines {
		lines[index] = fitLine(lines[index], width)
	}
	content := strings.Join(lines, "\n")
	if noColor {
		return content
	}
	return lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).Padding(1, 2).Foreground(lipgloss.Color("229")).BorderForeground(lipgloss.Color("214")).Render(content)
}

func Status(snapshot uit.Snapshot, busy bool, noColor bool, widths ...int) string {
	state := terminalSafe(string(snapshot.State))
	if busy {
		state += "/busy"
	}
	profile := terminalSafe(snapshot.ProfileLabel)
	if profile == "" {
		profile = "custom"
	}
	cache := ""
	if snapshot.Usage.PromptTokens > 0 {
		cache = fmt.Sprintf(" cache %.1f%%", snapshot.Usage.CacheHitPercent())
	}
	value := fmt.Sprintf("session %s  %s  profile %s  tokens %d%s  queued %d", terminalSafe(snapshotID(snapshot)), state, profile, snapshot.Usage.TotalTokens, cache, len(snapshot.Pending))
	if snapshot.Workspace != "" {
		value += "  " + terminalSafe(snapshot.Workspace)
	}
	if snapshot.Execution != "" {
		value += "  exec " + terminalSafe(snapshot.Execution)
	}
	value = fitLine(value, firstWidth(widths))
	if noColor {
		return value
	}
	return lipgloss.NewStyle().Foreground(lipgloss.Color("244")).Render(value)
}

func Footer(state uit.State, noColor bool, widths ...int) string {
	value := "enter send  shift+enter newline  alt+s steer  alt+f follow-up  alt+n next  ctrl+t thinking  ctrl+g tools  ctrl+c interrupt  /help"
	if state == uit.StateSuspended {
		value = "permission review active  a approve  d deny  enter resume  esc handoff"
	}
	value = fitLine(value, firstWidth(widths))
	if noColor {
		return value
	}
	return lipgloss.NewStyle().Foreground(lipgloss.Color("242")).Render(value)
}

func Banner(message string, failure bool, noColor bool, widths ...int) string {
	if message == "" {
		return ""
	}
	message = terminalSafe(message)
	if noColor && failure {
		message = "error: " + message
	}
	message = fitLine(message, firstWidth(widths))
	if noColor {
		return message
	}
	color := lipgloss.Color("42")
	if failure {
		color = lipgloss.Color("196")
	}
	return lipgloss.NewStyle().Foreground(color).Render(message)
}

func snapshotID(snapshot uit.Snapshot) string {
	if snapshot.SessionID != "" {
		return snapshot.SessionID
	}
	return "new"
}

func role(value string, noColor bool) string {
	value = strings.ToUpper(terminalSafe(value))
	if noColor {
		return value
	}
	return lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("75")).Render(value)
}

// terminalSafe removes C0/C1 controls from model-, host-, and capability-owned
// text before Bubble Tea or Lip Gloss sees it. Newlines and tabs remain useful
// transcript structure; escape, bell, carriage-return, and cursor controls can
// no longer become terminal instructions. Generated styling is applied only
// after this boundary.
func terminalSafe(value string) string {
	return strings.Map(func(current rune) rune {
		if current == '\n' || current == '\t' {
			return current
		}
		if unicode.IsControl(current) {
			return -1
		}
		return current
	}, value)
}

func boundedTerminalSafe(value string, limit int) string {
	value = terminalSafe(value)
	if limit <= 0 {
		return value
	}
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	return string(runes[:limit]) + "..."
}

func firstWidth(values []int) int {
	if len(values) == 0 {
		return 0
	}
	return values[0]
}

func fitLine(value string, width int) string {
	if width <= 0 || lipgloss.Width(value) <= width {
		return value
	}
	if width <= 3 {
		return strings.Repeat(".", width)
	}
	runes := []rune(value)
	for len(runes) > 0 && lipgloss.Width(string(runes)+"...") > width {
		runes = runes[:len(runes)-1]
	}
	return string(runes) + "..."
}
