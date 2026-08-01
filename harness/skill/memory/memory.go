// Package memory provides a concurrent in-process implementation of the skill
// Source port.
package memory

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"

	skillcore "github.com/regularkevvv/agentic/harness/skill"
)

type Config struct {
	MaxDescriptionBytes int
	MaxInstructionBytes int
	MaxResources        int
}

type Source struct {
	mu     sync.RWMutex
	skills map[skillcore.Scope]map[string]skillcore.Skill
	config Config
}

func New(config Config) (*Source, error) {
	if config.MaxDescriptionBytes == 0 {
		config.MaxDescriptionBytes = 4 << 10
	}
	if config.MaxInstructionBytes == 0 {
		config.MaxInstructionBytes = 128 << 10
	}
	if config.MaxResources == 0 {
		config.MaxResources = 100
	}
	if config.MaxDescriptionBytes <= 0 || config.MaxInstructionBytes <= 0 || config.MaxResources <= 0 {
		return nil, errors.New("skill memory limits must be positive")
	}
	return &Source{skills: make(map[skillcore.Scope]map[string]skillcore.Skill), config: config}, nil
}

func (s *Source) Put(scope skillcore.Scope, value skillcore.Skill) error {
	if err := skillcore.ValidateScope(scope); err != nil {
		return err
	}
	if err := skillcore.ValidateSkill(value, s.config.MaxDescriptionBytes, s.config.MaxInstructionBytes, s.config.MaxResources); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.skills[scope] == nil {
		s.skills[scope] = make(map[string]skillcore.Skill)
	}
	s.skills[scope][value.Name] = skillcore.Clone(value)
	return nil
}

func (s *Source) Delete(scope skillcore.Scope, name string) error {
	if err := skillcore.ValidateScope(scope); err != nil {
		return err
	}
	if err := skillcore.ValidateName(name); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.skills[scope] == nil {
		return fmt.Errorf("%w: %s", skillcore.ErrNotFound, name)
	}
	if _, ok := s.skills[scope][name]; !ok {
		return fmt.Errorf("%w: %s", skillcore.ErrNotFound, name)
	}
	delete(s.skills[scope], name)
	return nil
}

func (s *Source) List(ctx context.Context, scope skillcore.Scope, limit int) ([]skillcore.Descriptor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := skillcore.ValidateScope(scope); err != nil {
		return nil, err
	}
	if limit <= 0 {
		return nil, errors.New("skill list limit must be positive")
	}
	s.mu.RLock()
	values := make([]skillcore.Descriptor, 0, len(s.skills[scope]))
	for _, value := range s.skills[scope] {
		values = append(values, skillcore.Descriptor{Name: value.Name, Description: value.Description})
	}
	s.mu.RUnlock()
	sort.Slice(values, func(i, j int) bool { return values[i].Name < values[j].Name })
	if len(values) > limit {
		values = values[:limit]
	}
	return values, nil
}

func (s *Source) Read(ctx context.Context, scope skillcore.Scope, name string, maxBytes int) (skillcore.Skill, error) {
	if err := ctx.Err(); err != nil {
		return skillcore.Skill{}, err
	}
	if err := skillcore.ValidateScope(scope); err != nil {
		return skillcore.Skill{}, err
	}
	if err := skillcore.ValidateName(name); err != nil {
		return skillcore.Skill{}, err
	}
	if maxBytes <= 0 {
		return skillcore.Skill{}, errors.New("skill read bound must be positive")
	}
	s.mu.RLock()
	value, ok := s.skills[scope][name]
	s.mu.RUnlock()
	if !ok {
		return skillcore.Skill{}, fmt.Errorf("%w: %s", skillcore.ErrNotFound, name)
	}
	if len(value.Instructions) > maxBytes {
		return skillcore.Skill{}, fmt.Errorf("%w: %s instructions", skillcore.ErrLimitExceeded, name)
	}
	return skillcore.Clone(value), nil
}

var _ skillcore.Source = (*Source)(nil)
