// Package render contains deterministic, terminal-independent view assembly.
package render

import (
	"fmt"
	"strings"

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
		parts = append(parts, role("assistant", options.NoColor)+"\n"+preview+" ▌")
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
		lines = append(lines, entry.Text)
	}
	if options.Thinking != ThinkingHidden {
		for _, thinking := range entry.Thinking {
			if options.Thinking == ThinkingCollapsed {
				lines = append(lines, fmt.Sprintf("thinking (%d chars)", len([]rune(thinking.Text))))
			} else if thinking.Redacted {
				lines = append(lines, "thinking [redacted]")
			} else {
				lines = append(lines, "thinking: "+thinking.Text)
			}
		}
	}
	for _, tool := range entry.Tools {
		lines = append(lines, Tool(tool, options))
	}
	return strings.Join(lines, "\n")
}

func Tool(tool uit.Tool, options Options) string {
	name := tool.Name
	if name == "" {
		name = "tool"
	}
	result := fmt.Sprintf("[%s] %s", tool.State, name)
	if options.ToolExpanded && tool.Summary != "" {
		result += ": " + tool.Summary
	}
	return result
}

func Approval(value *uit.Suspension, selected int, decisions map[string]uit.DecisionAction, noColor bool) string {
	if value == nil {
		return ""
	}
	if !value.Supported {
		return "Suspended: " + value.Kind + "\n" + value.Description + "\nEsc quits safely."
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
		resource := approval.ResourceDisplay
		if resource == "" {
			resource = approval.CanonicalResource
		}
		lines = append(lines, fmt.Sprintf("%s[%s] %s: %s %s", cursor, mark, approval.ToolName, approval.Action, resource))
	}
	lines = append(lines, "a approve  d deny  ↑/↓ choose  enter resume  esc handoff")
	content := strings.Join(lines, "\n")
	if noColor {
		return content
	}
	return lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).Padding(1, 2).Foreground(lipgloss.Color("229")).BorderForeground(lipgloss.Color("214")).Render(content)
}

func Status(snapshot uit.Snapshot, busy bool, noColor bool) string {
	state := string(snapshot.State)
	if busy {
		state += "/busy"
	}
	profile := snapshot.ProfileLabel
	if profile == "" {
		profile = "custom"
	}
	cache := ""
	if snapshot.Usage.PromptTokens > 0 {
		cache = fmt.Sprintf(" cache %.1f%%", snapshot.Usage.CacheHitPercent())
	}
	value := fmt.Sprintf("session %s  %s  profile %s  tokens %d%s  queued %d", snapshotID(snapshot), state, profile, snapshot.Usage.TotalTokens, cache, len(snapshot.Pending))
	if snapshot.Workspace != "" {
		value += "  " + snapshot.Workspace
	}
	if noColor {
		return value
	}
	return lipgloss.NewStyle().Foreground(lipgloss.Color("244")).Render(value)
}

func Footer(state uit.State, noColor bool) string {
	value := "enter send  alt+s steer  alt+f follow-up  alt+n next  ctrl+t thinking  ctrl+g tools  ctrl+c interrupt  /help"
	if state == uit.StateSuspended {
		value = "permission review active  a approve  d deny  enter resume  esc handoff"
	}
	if noColor {
		return value
	}
	return lipgloss.NewStyle().Foreground(lipgloss.Color("242")).Render(value)
}

func Banner(message string, failure bool, noColor bool) string {
	if message == "" {
		return ""
	}
	if noColor {
		if failure {
			return "error: " + message
		}
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
	value = strings.ToUpper(value)
	if noColor {
		return value
	}
	return lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("75")).Render(value)
}
