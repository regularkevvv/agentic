// Package permission implements hierarchical, default-deny governance over
// canonical harness effects. It is policy, not an OS sandbox.
package permission

import (
	"errors"
	"fmt"
	"path"
	"strings"

	"github.com/regularkevvv/agentic/harness/env"
)

// Decision is the policy result for one proposed effect.
type Decision uint8

const (
	DecisionInvalid Decision = iota
	DecisionAllow
	DecisionAsk
	DecisionDeny
)

// Rule maps a slash-separated hierarchical pattern to one decision. "*" spans
// one segment and "**" spans the remaining suffix.
type Rule struct {
	Pattern  string
	Decision Decision
}

// PermissionRequest is the backend-neutral policy input described by the
// harness design.
type PermissionRequest struct {
	Capability        string
	Action            string
	CanonicalResource env.CanonicalResource
}

// Policy is an immutable most-specific-match ruleset.
type Policy struct {
	fallback Decision
	rules    []compiledRule
}

type compiledRule struct {
	rule        Rule
	segments    []string
	specificity int
	index       int
}

func New(fallback Decision, rules ...Rule) (*Policy, error) {
	if fallback != DecisionAllow && fallback != DecisionAsk && fallback != DecisionDeny {
		return nil, errors.New("permission fallback decision is invalid")
	}
	compiled := make([]compiledRule, len(rules))
	for index, rule := range rules {
		if rule.Decision != DecisionAllow && rule.Decision != DecisionAsk && rule.Decision != DecisionDeny {
			return nil, fmt.Errorf("permission rule %d has an invalid decision", index)
		}
		segments, err := patternSegments(rule.Pattern)
		if err != nil {
			return nil, fmt.Errorf("permission rule %d: %w", index, err)
		}
		specificity := 0
		for _, segment := range segments {
			for _, char := range segment {
				if char != '*' && char != '?' && char != '[' && char != ']' {
					specificity++
				}
			}
		}
		compiled[index] = compiledRule{rule: rule, segments: segments, specificity: specificity, index: index}
	}
	return &Policy{fallback: fallback, rules: compiled}, nil
}

// Evaluate applies the most-specific matching rule. Later rules win exact
// specificity ties, which makes narrow application overrides explicit.
func (p *Policy) Evaluate(request PermissionRequest) Decision {
	if p == nil {
		return DecisionDeny
	}
	segments := requestSegments(request)
	decision := p.fallback
	bestSpecificity := -1
	bestIndex := -1
	for _, rule := range p.rules {
		if !matchPattern(rule.segments, segments) {
			continue
		}
		if rule.specificity > bestSpecificity ||
			(rule.specificity == bestSpecificity && rule.index > bestIndex) {
			decision = rule.rule.Decision
			bestSpecificity = rule.specificity
			bestIndex = rule.index
		}
	}
	return decision
}

func patternSegments(pattern string) ([]string, error) {
	pattern = strings.Trim(pattern, "/")
	if pattern == "" {
		return nil, errors.New("permission pattern is empty")
	}
	segments := strings.Split(pattern, "/")
	for index, segment := range segments {
		if segment == "" {
			return nil, errors.New("permission pattern contains an empty segment")
		}
		if segment == "**" && index != len(segments)-1 {
			return nil, errors.New("** is allowed only as the final segment")
		}
		if segment != "**" {
			if _, err := path.Match(segment, "probe"); err != nil {
				return nil, fmt.Errorf("invalid segment %q: %w", segment, err)
			}
		}
	}
	return segments, nil
}

func requestSegments(request PermissionRequest) []string {
	result := []string{request.Capability, request.Action}
	if request.CanonicalResource.Scheme != "" {
		result = append(result, request.CanonicalResource.Scheme)
	}
	id := strings.Trim(strings.ReplaceAll(request.CanonicalResource.ID, "\\", "/"), "/")
	if id != "" {
		result = append(result, strings.Split(id, "/")...)
	}
	return result
}

func matchPattern(pattern, value []string) bool {
	for index, segment := range pattern {
		if segment == "**" {
			return true
		}
		if index >= len(value) {
			return false
		}
		matched, err := path.Match(segment, value[index])
		if err != nil || !matched {
			return false
		}
	}
	return len(pattern) == len(value)
}
