// Package render contains deterministic, terminal-independent view assembly.
package render

import (
	"fmt"
	"image/color"
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
	for _, entry := range foldToolActivity(entries) {
		if entry.Text == "" && len(entry.Thinking) == 0 && len(entry.Tools) == 0 {
			continue
		}
		parts = append(parts, Entry(entry, options))
	}
	if preview != "" {
		cursor := " ▌"
		if options.NoColor {
			cursor = " |"
		}
		parts = append(parts, role("assistant", options.NoColor)+"\n"+markdown(preview, options.NoColor)+cursor)
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
	toolOnly := entry.Text == "" && len(entry.Thinking) == 0 && len(entry.Tools) > 0
	lines := make([]string, 0, 4)
	if !toolOnly {
		lines = append(lines, role(label, options.NoColor))
	}
	if options.Thinking != ThinkingHidden {
		for _, thinking := range entry.Thinking {
			if options.Thinking == ThinkingCollapsed {
				lines = append(lines, muted(fmt.Sprintf("thinking collapsed (%d chars) · ctrl+t to show", len([]rune(thinking.Text))), options.NoColor))
			} else if thinking.Redacted {
				lines = append(lines, muted("thinking [redacted]", options.NoColor))
			} else {
				lines = append(lines, muted("thinking", options.NoColor)+"\n"+indent(markdown(thinking.Text, options.NoColor), "  "))
			}
		}
	}
	if entry.Text != "" {
		lines = append(lines, markdown(entry.Text, options.NoColor))
	}
	if len(entry.Tools) > 0 {
		lines = append(lines, Tools(entry.Tools, options))
	}
	return strings.Join(lines, "\n")
}

func Tool(tool uit.Tool, options Options) string {
	title := boundedTerminalSafe(tool.Presentation.Title, 256)
	if title == "" {
		title = humanize(tool.Name)
	}
	result := toolGlyph(tool.State) + " " + title
	detail := toolDetail(tool)
	if options.ToolExpanded && detail != "" {
		result += "\n  " + muted(boundedTerminalSafe(detail, 512), options.NoColor)
	}
	if options.NoColor {
		return result
	}
	return lipgloss.NewStyle().Foreground(toolColor(tool.State)).Render(result)
}

// Tools groups activity using host-supplied semantic categories while keeping
// the individual safe labels visible. Raw arguments and results never reach
// this renderer.
func Tools(tools []uit.Tool, options Options) string {
	type group struct {
		category uit.ToolCategory
		tools    []uit.Tool
	}
	groups := make([]group, 0, len(tools))
	for _, tool := range tools {
		category := tool.Presentation.Category
		if category == "" {
			category = inferCategory(tool.Name)
		}
		if len(groups) == 0 || groups[len(groups)-1].category != category {
			groups = append(groups, group{category: category})
		}
		groups[len(groups)-1].tools = append(groups[len(groups)-1].tools, tool)
	}
	blocks := make([]string, 0, len(groups))
	for _, current := range groups {
		state := aggregateToolState(current.tools)
		header := toolGlyph(state) + " " + toolGroupLabel(current.category, state)
		if !options.NoColor {
			header = lipgloss.NewStyle().Bold(true).Foreground(toolColor(state)).Render(header)
		}
		if len(current.tools) == 1 {
			tool := current.tools[0]
			title := boundedTerminalSafe(tool.Presentation.Title, 256)
			if title == "" {
				title = humanize(tool.Name)
			}
			line := header + " " + title
			if detail := toolDetail(tool); options.ToolExpanded && detail != "" {
				line += "\n  " + muted(boundedTerminalSafe(detail, 512), options.NoColor)
			}
			blocks = append(blocks, line)
			continue
		}
		lines := []string{header}
		for index, tool := range current.tools {
			branch := "├─"
			if index == len(current.tools)-1 {
				branch = "└─"
			}
			title := boundedTerminalSafe(tool.Presentation.Title, 256)
			if title == "" {
				title = humanize(tool.Name)
			}
			line := "  " + branch + " " + title
			if !options.NoColor {
				line = lipgloss.NewStyle().Foreground(lipgloss.Color("252")).Render(line)
			}
			lines = append(lines, line)
			detail := toolDetail(tool)
			if options.ToolExpanded && detail != "" {
				lines = append(lines, "     "+muted(boundedTerminalSafe(detail, 512), options.NoColor))
			}
		}
		blocks = append(blocks, strings.Join(lines, "\n"))
	}
	return strings.Join(blocks, "\n")
}

type toolPosition struct{ entry, tool int }

func toolDetail(tool uit.Tool) string {
	if tool.Presentation.Detail != "" {
		return tool.Presentation.Detail
	}
	if tool.Presentation.Title == "" {
		return tool.Summary
	}
	return ""
}

// foldToolActivity collapses planned/running/result lifecycle records by call
// ID, preserves the richest safe presentation, and coalesces adjacent tool-only
// transcript entries into one activity block. Assistant text remains a hard
// grouping boundary so commentary keeps its conversational position.
func foldToolActivity(entries []uit.Entry) []uit.Entry {
	latest := latestToolPositions(entries)
	merged := make(map[string]uit.Tool)
	for _, entry := range entries {
		for _, tool := range entry.Tools {
			if tool.CallID == "" {
				continue
			}
			merged[tool.CallID] = mergeTool(merged[tool.CallID], tool)
		}
	}
	result := make([]uit.Entry, 0, len(entries))
	pending := make([]uit.Tool, 0)
	flush := func() {
		if len(pending) == 0 {
			return
		}
		result = append(result, uit.Entry{Role: uit.RoleTool, Tools: pending})
		pending = nil
	}
	for entryIndex, entry := range entries {
		entry.Tools = visibleTools(entry.Tools, entryIndex, latest)
		for index, tool := range entry.Tools {
			if tool.CallID != "" {
				entry.Tools[index] = merged[tool.CallID]
			}
		}
		toolOnly := entry.Text == "" && len(entry.Thinking) == 0 && len(entry.Tools) > 0
		if toolOnly {
			pending = append(pending, entry.Tools...)
			continue
		}
		flush()
		if entry.Text != "" || len(entry.Thinking) > 0 || len(entry.Tools) > 0 {
			result = append(result, entry)
		}
	}
	flush()
	return result
}

func mergeTool(previous, current uit.Tool) uit.Tool {
	result := current
	if result.Name == "" {
		result.Name = previous.Name
	}
	if result.Attempt == 0 {
		result.Attempt = previous.Attempt
	}
	if result.Summary == "" {
		result.Summary = previous.Summary
		if previous.Summary != "" {
			result.Presentation = previous.Presentation
		}
	}
	if result.Presentation.Category == "" {
		result.Presentation.Category = previous.Presentation.Category
	}
	if result.Presentation.Title == "" {
		result.Presentation.Title = previous.Presentation.Title
	}
	if result.Presentation.Detail == "" {
		result.Presentation.Detail = previous.Presentation.Detail
	}
	return result
}

func latestToolPositions(entries []uit.Entry) map[string]toolPosition {
	result := make(map[string]toolPosition)
	for entryIndex, entry := range entries {
		for toolIndex, tool := range entry.Tools {
			if tool.CallID != "" {
				result[tool.CallID] = toolPosition{entry: entryIndex, tool: toolIndex}
			}
		}
	}
	return result
}

func visibleTools(tools []uit.Tool, entry int, latest map[string]toolPosition) []uit.Tool {
	result := make([]uit.Tool, 0, len(tools))
	for index, tool := range tools {
		if tool.CallID != "" && latest[tool.CallID] != (toolPosition{entry: entry, tool: index}) {
			continue
		}
		result = append(result, tool)
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

func ApprovalResolving(noColor bool, widths ...int) string {
	value := "Applying permission decisions..."
	if !noColor {
		value = "◌ " + value
		value = lipgloss.NewStyle().Foreground(lipgloss.Color("244")).Render(value)
	}
	return fitLine(value, firstWidth(widths))
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
	value := "enter send  shift+enter newline  pgup/pgdn scroll  alt+s steer  alt+f follow-up  alt+n next  ctrl+t thinking  ctrl+g tools  ctrl+c interrupt  /help"
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
	color := lipgloss.Color("244")
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

func markdown(value string, noColor bool) string {
	value = terminalSafe(value)
	lines := strings.Split(value, "\n")
	fenced := false
	for index, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			fenced = !fenced
			language := strings.TrimSpace(strings.TrimPrefix(trimmed, "```"))
			if language == "" {
				lines[index] = muted("│", noColor)
			} else {
				lines[index] = muted("│ "+language, noColor)
			}
			continue
		}
		if fenced {
			lines[index] = muted("│ ", noColor) + inlineMarkdown(line, noColor)
			continue
		}
		if heading := strings.TrimLeft(line, "#"); len(heading) < len(line) && strings.HasPrefix(heading, " ") {
			text := inlineMarkdown(strings.TrimSpace(heading), noColor)
			if !noColor {
				text = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("75")).Render(text)
			}
			lines[index] = text
			continue
		}
		if strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "* ") {
			indentation := line[:len(line)-len(strings.TrimLeft(line, " \t"))]
			lines[index] = indentation + "• " + inlineMarkdown(strings.TrimSpace(trimmed[2:]), noColor)
			continue
		}
		if strings.HasPrefix(trimmed, "> ") {
			lines[index] = muted("│ ", noColor) + inlineMarkdown(strings.TrimSpace(trimmed[2:]), noColor)
			continue
		}
		lines[index] = inlineMarkdown(line, noColor)
	}
	return strings.Join(lines, "\n")
}

func inlineMarkdown(value string, noColor bool) string {
	var result strings.Builder
	for len(value) > 0 {
		switch {
		case strings.HasPrefix(value, "**"):
			end := strings.Index(value[2:], "**")
			if end < 0 {
				result.WriteString(value)
				return result.String()
			}
			content := value[2 : end+2]
			if noColor {
				result.WriteString(content)
			} else {
				result.WriteString(lipgloss.NewStyle().Bold(true).Render(content))
			}
			value = value[end+4:]
		case value[0] == '`':
			end := strings.IndexByte(value[1:], '`')
			if end < 0 {
				result.WriteString(value)
				return result.String()
			}
			content := value[1 : end+1]
			if noColor {
				result.WriteString(content)
			} else {
				result.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color("81")).Render(content))
			}
			value = value[end+2:]
		default:
			result.WriteByte(value[0])
			value = value[1:]
		}
	}
	return result.String()
}

func muted(value string, noColor bool) string {
	value = terminalSafe(value)
	if noColor {
		return value
	}
	return lipgloss.NewStyle().Foreground(lipgloss.Color("244")).Render(value)
}

func indent(value, prefix string) string {
	return prefix + strings.ReplaceAll(value, "\n", "\n"+prefix)
}

func humanize(value string) string {
	value = boundedTerminalSafe(value, 128)
	if value == "" {
		return "Tool"
	}
	value = strings.ReplaceAll(value, "_", " ")
	runes := []rune(value)
	runes[0] = unicode.ToUpper(runes[0])
	return string(runes)
}

func inferCategory(name string) uit.ToolCategory {
	switch name {
	case "read_file", "list_files", "stat_file", "read_artifact":
		return uit.ToolCategoryExplore
	case "run_command":
		return uit.ToolCategoryExecute
	case "write_file", "make_directory", "remove_path":
		return uit.ToolCategoryChange
	default:
		return uit.ToolCategoryOther
	}
}

func aggregateToolState(tools []uit.Tool) uit.ToolState {
	state := uit.ToolDone
	for _, tool := range tools {
		switch tool.State {
		case uit.ToolError:
			return uit.ToolError
		case uit.ToolRunning:
			state = uit.ToolRunning
		case uit.ToolPreview, uit.ToolPlanned:
			if state == uit.ToolDone {
				state = uit.ToolPlanned
			}
		}
	}
	return state
}

func toolGroupLabel(category uit.ToolCategory, state uit.ToolState) string {
	labels := map[uit.ToolCategory][3]string{
		uit.ToolCategoryExplore: {"Exploring", "Explored", "Exploration failed"},
		uit.ToolCategoryExecute: {"Running", "Ran", "Command failed"},
		uit.ToolCategoryChange:  {"Changing", "Changed", "Change failed"},
		uit.ToolCategoryOther:   {"Using tools", "Used tools", "Tool failed"},
	}
	values, ok := labels[category]
	if !ok {
		values = labels[uit.ToolCategoryOther]
	}
	if state == uit.ToolError {
		return values[2]
	}
	if state == uit.ToolDone {
		return values[1]
	}
	return values[0]
}

func toolGlyph(state uit.ToolState) string {
	switch state {
	case uit.ToolDone:
		return "✓"
	case uit.ToolError:
		return "✗"
	case uit.ToolRunning:
		return "•"
	default:
		return "○"
	}
}

func toolColor(state uit.ToolState) color.Color {
	switch state {
	case uit.ToolDone:
		return lipgloss.Color("42")
	case uit.ToolError:
		return lipgloss.Color("196")
	case uit.ToolRunning:
		return lipgloss.Color("75")
	default:
		return lipgloss.Color("244")
	}
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
