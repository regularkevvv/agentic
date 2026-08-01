// Package filesystem discovers bounded SKILL.md directories behind the skill
// Source port. It returns logical names only and rejects canonical-path escape.
package filesystem

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"

	skillcore "github.com/regularkevvv/agentic/harness/skill"
)

type Config struct {
	Root                string
	MaxDescriptionBytes int
	MaxResources        int
}

type Source struct {
	root         string
	maxDesc      int
	maxResources int
}

func New(config Config) (*Source, error) {
	if config.Root == "" {
		return nil, errors.New("skill filesystem root is required")
	}
	abs, err := filepath.Abs(config.Root)
	if err != nil {
		return nil, fmt.Errorf("canonicalize skill root: %w", err)
	}
	real, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return nil, fmt.Errorf("resolve skill root: %w", err)
	}
	info, err := os.Stat(real)
	if err != nil || !info.IsDir() {
		if err == nil {
			err = errors.New("not a directory")
		}
		return nil, fmt.Errorf("inspect skill root: %w", err)
	}
	if config.MaxDescriptionBytes == 0 {
		config.MaxDescriptionBytes = 4 << 10
	}
	if config.MaxResources == 0 {
		config.MaxResources = 100
	}
	if config.MaxDescriptionBytes <= 0 || config.MaxResources <= 0 {
		return nil, errors.New("skill filesystem limits must be positive")
	}
	return &Source{root: filepath.Clean(real), maxDesc: config.MaxDescriptionBytes, maxResources: config.MaxResources}, nil
}

func (s *Source) Root() string { return s.root }

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
	directory, err := s.scopeDirectory(scope)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, err
	}
	values := make([]skillcore.Descriptor, 0)
	seen := make(map[string]bool)
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := skillcore.ValidateName(entry.Name()); err != nil {
			continue
		}
		skillDir, err := canonicalInside(directory, filepath.Join(directory, entry.Name()))
		if err != nil {
			return nil, err
		}
		info, err := os.Stat(skillDir)
		if err != nil || !info.IsDir() {
			if err != nil {
				return nil, err
			}
			continue
		}
		manifest, err := canonicalInside(skillDir, filepath.Join(skillDir, "SKILL.md"))
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		name, description, err := readFrontmatter(manifest, s.maxDesc)
		if err != nil {
			return nil, fmt.Errorf("read skill %s: %w", entry.Name(), err)
		}
		if name != entry.Name() {
			return nil, fmt.Errorf("skill directory %q declares name %q", entry.Name(), name)
		}
		if seen[name] {
			return nil, fmt.Errorf("duplicate skill %q", name)
		}
		seen[name] = true
		values = append(values, skillcore.Descriptor{Name: name, Description: description})
	}
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
	directory, err := s.scopeDirectory(scope)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return skillcore.Skill{}, fmt.Errorf("%w: %s", skillcore.ErrNotFound, name)
		}
		return skillcore.Skill{}, err
	}
	skillDir, err := canonicalInside(directory, filepath.Join(directory, name))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return skillcore.Skill{}, fmt.Errorf("%w: %s", skillcore.ErrNotFound, name)
		}
		return skillcore.Skill{}, err
	}
	manifest, err := canonicalInside(skillDir, filepath.Join(skillDir, "SKILL.md"))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return skillcore.Skill{}, fmt.Errorf("%w: %s", skillcore.ErrNotFound, name)
		}
		return skillcore.Skill{}, err
	}
	data, err := readBounded(manifest, maxBytes+s.maxDesc+(16<<10))
	if err != nil {
		return skillcore.Skill{}, err
	}
	declared, description, instructions, err := parseManifest(data, s.maxDesc, maxBytes)
	if err != nil {
		return skillcore.Skill{}, err
	}
	if declared != name {
		return skillcore.Skill{}, fmt.Errorf("skill directory %q declares name %q", name, declared)
	}
	resources, err := s.resources(skillDir)
	if err != nil {
		return skillcore.Skill{}, err
	}
	return skillcore.Skill{Name: name, Description: description, Instructions: instructions, Resources: resources}, nil
}

func (s *Source) scopeDirectory(scope skillcore.Scope) (string, error) {
	name := url.PathEscape(string(scope))
	entries, err := os.ReadDir(s.root)
	if err != nil {
		return "", err
	}
	found := false
	for _, entry := range entries {
		if entry.Name() == name {
			found = true
			break
		}
	}
	if !found {
		return "", fs.ErrNotExist
	}
	candidate := filepath.Clean(filepath.Join(s.root, name))
	real, err := canonicalInside(s.root, candidate)
	if err != nil {
		return "", err
	}
	if real != candidate {
		return "", errors.New("skill scope path resolves through a symlink")
	}
	return real, nil
}

func (s *Source) resources(skillDir string) ([]string, error) {
	entries, err := os.ReadDir(skillDir)
	if err != nil {
		return nil, err
	}
	resources := make([]string, 0)
	for _, entry := range entries {
		if entry.Name() == "SKILL.md" {
			continue
		}
		if _, err := canonicalInside(skillDir, filepath.Join(skillDir, entry.Name())); err != nil {
			return nil, err
		}
		if strings.ContainsAny(entry.Name(), "\x00\r\n") {
			return nil, errors.New("invalid skill resource name")
		}
		resources = append(resources, entry.Name())
	}
	sort.Strings(resources)
	if len(resources) > s.maxResources {
		return nil, fmt.Errorf("%w: too many resources", skillcore.ErrLimitExceeded)
	}
	return resources, nil
}

func canonicalInside(base, candidate string) (string, error) {
	real, err := filepath.EvalSymlinks(candidate)
	if err != nil {
		return "", err
	}
	relative, err := filepath.Rel(base, real)
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) || filepath.IsAbs(relative) {
		return "", errors.New("skill path escapes its canonical scope")
	}
	return filepath.Clean(real), nil
}

func readFrontmatter(path string, maxDescription int) (string, string, error) {
	data, err := readBounded(path, maxDescription+(16<<10))
	if err != nil && !errors.Is(err, skillcore.ErrLimitExceeded) {
		return "", "", err
	}
	name, description, _, parseErr := parseManifest(data, maxDescription, 1<<30)
	return name, description, parseErr
}

func readBounded(path string, maximum int) (data []byte, err error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		if closeErr := file.Close(); err == nil {
			err = closeErr
		}
	}()
	data, err = io.ReadAll(io.LimitReader(file, int64(maximum)+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maximum {
		return data, skillcore.ErrLimitExceeded
	}
	return data, nil
}

func parseManifest(data []byte, maxDescription, maxInstructions int) (string, string, string, error) {
	normalized := bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))
	if !bytes.HasPrefix(normalized, []byte("---\n")) {
		return "", "", "", errors.New("SKILL.md requires frontmatter")
	}
	end := bytes.Index(normalized[4:], []byte("\n---\n"))
	if end < 0 {
		return "", "", "", errors.New("SKILL.md frontmatter is not terminated")
	}
	end += 4
	var name, description string
	for _, line := range strings.Split(string(normalized[4:end]), "\n") {
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		switch strings.TrimSpace(key) {
		case "name":
			if name != "" {
				return "", "", "", errors.New("duplicate skill name frontmatter")
			}
			name = value
		case "description":
			if description != "" {
				return "", "", "", errors.New("duplicate skill description frontmatter")
			}
			description = value
		}
	}
	if err := skillcore.ValidateDescriptor(skillcore.Descriptor{Name: name, Description: description}, maxDescription); err != nil {
		return "", "", "", err
	}
	instructions := strings.TrimSpace(string(normalized[end+5:]))
	if instructions == "" || len(instructions) > maxInstructions {
		return "", "", "", fmt.Errorf("%w: instructions", skillcore.ErrLimitExceeded)
	}
	return name, description, instructions, nil
}

var _ skillcore.Source = (*Source)(nil)
