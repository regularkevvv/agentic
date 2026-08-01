// Package skill defines bounded application-supplied skill source ports and
// an opt-in harness capability. Sources, not the capability, decide storage.
package skill

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
)

var (
	ErrNotFound      = errors.New("skill not found")
	ErrInvalidName   = errors.New("invalid skill name")
	ErrLimitExceeded = errors.New("skill bound exceeded")
)

var namePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$`)

type Scope string

type Descriptor struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

type Skill struct {
	Name         string   `json:"name"`
	Description  string   `json:"description"`
	Instructions string   `json:"instructions"`
	Resources    []string `json:"resources,omitempty"`
}

type Source interface {
	List(context.Context, Scope, int) ([]Descriptor, error)
	Read(context.Context, Scope, string, int) (Skill, error)
}

func ValidateScope(scope Scope) error {
	value := string(scope)
	if value == "" || len(value) > 256 || strings.TrimSpace(value) != value || strings.ContainsAny(value, "\x00\r\n") {
		return errors.New("invalid skill scope")
	}
	return nil
}

func ValidateName(name string) error {
	if !namePattern.MatchString(name) {
		return fmt.Errorf("%w: %q", ErrInvalidName, name)
	}
	return nil
}

func ValidateDescriptor(value Descriptor, maxDescriptionBytes int) error {
	if err := ValidateName(value.Name); err != nil {
		return err
	}
	if value.Description == "" || len(value.Description) > maxDescriptionBytes {
		return fmt.Errorf("%w: description for %s", ErrLimitExceeded, value.Name)
	}
	return nil
}

func ValidateSkill(value Skill, maxDescriptionBytes, maxInstructionBytes, maxResources int) error {
	if err := ValidateDescriptor(Descriptor{Name: value.Name, Description: value.Description}, maxDescriptionBytes); err != nil {
		return err
	}
	if value.Instructions == "" || len(value.Instructions) > maxInstructionBytes || len(value.Resources) > maxResources {
		return fmt.Errorf("%w: skill %s", ErrLimitExceeded, value.Name)
	}
	seen := make(map[string]bool, len(value.Resources))
	for _, resource := range value.Resources {
		if resource == "" || len(resource) > 256 || strings.ContainsAny(resource, "\x00\r\n") || seen[resource] {
			return errors.New("invalid or duplicate skill resource name")
		}
		seen[resource] = true
	}
	return nil
}

func Clone(value Skill) Skill {
	value.Resources = append([]string(nil), value.Resources...)
	return value
}
