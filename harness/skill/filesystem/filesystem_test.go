package filesystem

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	skillcore "github.com/regularkevvv/agentic/harness/skill"
)

func writeSkill(t *testing.T, root, scope, directory, name, description, instructions string) string {
	t.Helper()
	path := filepath.Join(root, scope, directory)
	if err := os.MkdirAll(path, 0o700); err != nil {
		t.Fatal(err)
	}
	manifest := "---\nname: " + name + "\ndescription: " + description + "\n---\n" + instructions
	if err := os.WriteFile(filepath.Join(path, "SKILL.md"), []byte(manifest), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestFilesystemOrderingBoundsResourcesAndIsolation(t *testing.T) {
	root := t.TempDir()
	beta := writeSkill(t, root, "team", "beta", "beta", "second", "beta instructions")
	alpha := writeSkill(t, root, "team", "alpha", "alpha", "first", "alpha instructions")
	writeSkill(t, root, "other", "alpha", "alpha", "other", "isolated")
	if err := os.WriteFile(filepath.Join(alpha, "example.txt"), []byte("resource"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(alpha, "assets"), 0o700); err != nil {
		t.Fatal(err)
	}
	_ = beta
	source, err := New(Config{Root: root, MaxDescriptionBytes: 32, MaxResources: 3})
	if err != nil {
		t.Fatal(err)
	}
	values, err := source.List(context.Background(), "team", 10)
	if err != nil || !reflect.DeepEqual(values, []skillcore.Descriptor{{Name: "alpha", Description: "first"}, {Name: "beta", Description: "second"}}) {
		t.Fatalf("list = %#v, %v", values, err)
	}
	one, _ := source.List(context.Background(), "team", 1)
	if len(one) != 1 || one[0].Name != "alpha" {
		t.Fatalf("bounded list = %#v", one)
	}
	value, err := source.Read(context.Background(), "team", "alpha", 64)
	if err != nil || value.Instructions != "alpha instructions" || !reflect.DeepEqual(value.Resources, []string{"assets", "example.txt"}) {
		t.Fatalf("read = %#v, %v", value, err)
	}
	if _, err := source.Read(context.Background(), "team", "alpha", 5); !errors.Is(err, skillcore.ErrLimitExceeded) {
		t.Fatalf("instruction bound = %v", err)
	}
	isolated, err := source.Read(context.Background(), "other", "alpha", 64)
	if err != nil || isolated.Description != "other" {
		t.Fatalf("isolated = %#v, %v", isolated, err)
	}
	missing, err := source.List(context.Background(), "missing", 2)
	if err != nil || len(missing) != 0 {
		t.Fatalf("missing scope = %#v, %v", missing, err)
	}
	if !filepath.IsAbs(source.Root()) {
		t.Fatalf("source root is not canonical: %q", source.Root())
	}
}

func TestFilesystemRejectsSymlinkEscapeAndMalformedManifests(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	writeSkill(t, outside, ".", "evil", "evil", "escape", "outside")
	if err := os.MkdirAll(filepath.Join(root, "team"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(outside, "evil"), filepath.Join(root, "team", "evil")); err != nil {
		t.Fatal(err)
	}
	source, _ := New(Config{Root: root})
	if _, err := source.List(context.Background(), "team", 10); err == nil || !stringsContains(err.Error(), "escapes") {
		t.Fatalf("symlink escape = %v", err)
	}

	root = t.TempDir()
	writeSkill(t, root, "team", "wrong", "different", "bad", "body")
	source, _ = New(Config{Root: root})
	if _, err := source.List(context.Background(), "team", 10); err == nil {
		t.Fatal("mismatched frontmatter succeeded")
	}
	manifest := filepath.Join(root, "team", "wrong", "SKILL.md")
	if err := os.WriteFile(manifest, []byte("no frontmatter"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := source.Read(context.Background(), "team", "wrong", 100); err == nil {
		t.Fatal("malformed manifest succeeded")
	}
}

func TestFilesystemValidation(t *testing.T) {
	if _, err := New(Config{}); err == nil {
		t.Fatal("missing root succeeded")
	}
	file := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(file, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := New(Config{Root: file}); err == nil {
		t.Fatal("file root succeeded")
	}
	root := t.TempDir()
	if _, err := New(Config{Root: root, MaxResources: -1}); err == nil {
		t.Fatal("negative limits succeeded")
	}
	source, _ := New(Config{Root: root})
	if _, err := source.List(context.Background(), "", 1); err == nil {
		t.Fatal("invalid scope succeeded")
	}
	if _, err := source.Read(context.Background(), "team", "../bad", 1); err == nil {
		t.Fatal("invalid name succeeded")
	}
}

func TestFilesystemCancellationMissingEntriesAndResourceBounds(t *testing.T) {
	if _, err := New(filepathConfig(filepath.Join(t.TempDir(), "missing"))); err == nil {
		t.Fatal("missing root succeeded")
	}
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "team", "no-manifest"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "team", "plain-file"), []byte("not a directory"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "team", ".invalid"), []byte("ignored"), 0o600); err != nil {
		t.Fatal(err)
	}
	skillDir := writeSkill(t, root, "team", "bounded", "bounded", "description", strings.Repeat("x", 20<<10))
	if err := os.WriteFile(filepath.Join(skillDir, "one"), []byte("1"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(skillDir, "two"), []byte("2"), 0o600); err != nil {
		t.Fatal(err)
	}
	source, err := New(Config{Root: root, MaxDescriptionBytes: 32, MaxResources: 1})
	if err != nil {
		t.Fatal(err)
	}
	values, err := source.List(context.Background(), "team", 10)
	if err != nil || len(values) != 1 || values[0].Name != "bounded" {
		t.Fatalf("filtered list = %#v, %v", values, err)
	}
	if _, err := source.Read(context.Background(), "team", "bounded", 21<<10); !errors.Is(err, skillcore.ErrLimitExceeded) {
		t.Fatalf("resource bound = %v", err)
	}
	if _, err := source.Read(context.Background(), "missing", "bounded", 10); !errors.Is(err, skillcore.ErrNotFound) {
		t.Fatalf("missing scope = %v", err)
	}
	if _, err := source.Read(context.Background(), "team", "missing", 10); !errors.Is(err, skillcore.ErrNotFound) {
		t.Fatalf("missing skill = %v", err)
	}
	if _, err := source.Read(context.Background(), "team", "no-manifest", 10); !errors.Is(err, skillcore.ErrNotFound) {
		t.Fatalf("missing manifest = %v", err)
	}
	if _, err := source.Read(context.Background(), "team", "bounded", 0); err == nil {
		t.Fatal("zero read bound succeeded")
	}
	if _, err := source.List(context.Background(), "team", 0); err == nil {
		t.Fatal("zero list bound succeeded")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := source.List(canceled, "team", 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled list = %v", err)
	}
	if _, err := source.Read(canceled, "team", "bounded", 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled read = %v", err)
	}
	if _, err := source.Read(context.Background(), "", "bounded", 1); err == nil {
		t.Fatal("invalid read scope succeeded")
	}
}

func TestFilesystemResourceSymlinkAndDirectHelpers(t *testing.T) {
	root := t.TempDir()
	skillDir := writeSkill(t, root, "team", "safe", "safe", "description", "instructions")
	outside := filepath.Join(t.TempDir(), "outside.txt")
	if err := os.WriteFile(outside, []byte("outside"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(skillDir, "escape")); err != nil {
		t.Fatal(err)
	}
	source, _ := New(Config{Root: root})
	if _, err := source.Read(context.Background(), "team", "safe", 100); err == nil || !stringsContains(err.Error(), "escapes") {
		t.Fatalf("resource symlink escape = %v", err)
	}
	if _, err := source.resources(filepath.Join(root, "missing")); err == nil {
		t.Fatal("resources on missing directory succeeded")
	}
	file := filepath.Join(t.TempDir(), "bounded")
	if err := os.WriteFile(file, []byte("12345"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readBounded(file, 4); !errors.Is(err, skillcore.ErrLimitExceeded) {
		t.Fatalf("read bound = %v", err)
	}
	if _, err := readBounded(filepath.Join(t.TempDir(), "missing"), 1); err == nil {
		t.Fatal("read missing file succeeded")
	}
	if _, err := canonicalInside(root, filepath.Join(root, "missing")); err == nil {
		t.Fatal("canonicalized missing path")
	}
}

func TestManifestParserFrontiers(t *testing.T) {
	validCRLF := []byte("---\r\nname: quoted\r\ndescription: \"value\"\r\nignored line\r\n---\r\nbody\r\n")
	name, description, instructions, err := parseManifest(validCRLF, 20, 20)
	if err != nil || name != "quoted" || description != "value" || instructions != "body" {
		t.Fatalf("CRLF manifest = %q %q %q, %v", name, description, instructions, err)
	}
	for name, data := range map[string][]byte{
		"no frontmatter":        []byte("body"),
		"unterminated":          []byte("---\nname: one\nbody"),
		"duplicate name":        []byte("---\nname: one\nname: two\ndescription: desc\n---\nbody"),
		"duplicate description": []byte("---\nname: one\ndescription: a\ndescription: b\n---\nbody"),
		"missing description":   []byte("---\nname: one\n---\nbody"),
		"empty instructions":    []byte("---\nname: one\ndescription: desc\n---\n   "),
		"long instructions":     []byte("---\nname: one\ndescription: desc\n---\nlong"),
	} {
		t.Run(name, func(t *testing.T) {
			maximum := 20
			if name == "long instructions" {
				maximum = 2
			}
			if _, _, _, err := parseManifest(data, 20, maximum); err == nil {
				t.Fatal("invalid manifest succeeded")
			}
		})
	}
}

func TestFilesystemCanonicalAndIOErrorFrontiers(t *testing.T) {
	t.Run("scope symlink escape", func(t *testing.T) {
		root := t.TempDir()
		outside := t.TempDir()
		if err := os.Symlink(outside, filepath.Join(root, "team")); err != nil {
			t.Fatal(err)
		}
		source, _ := New(Config{Root: root})
		if _, err := source.List(context.Background(), "team", 1); err == nil || !stringsContains(err.Error(), "escapes") {
			t.Fatalf("list scope escape = %v", err)
		}
		if _, err := source.Read(context.Background(), "team", "skill", 1); err == nil || !stringsContains(err.Error(), "escapes") {
			t.Fatalf("read scope escape = %v", err)
		}
	})

	t.Run("scope symlink alias", func(t *testing.T) {
		root := t.TempDir()
		writeSkill(t, root, "other", "skill", "skill", "value", "body")
		if err := os.Symlink(filepath.Join(root, "other"), filepath.Join(root, "team")); err != nil {
			t.Fatal(err)
		}
		source, _ := New(Config{Root: root})
		if _, err := source.List(context.Background(), "team", 1); err == nil || !stringsContains(err.Error(), "symlink") {
			t.Fatalf("scope alias = %v", err)
		}
		if _, err := source.Read(context.Background(), "team", "skill", 100); err == nil || !stringsContains(err.Error(), "symlink") {
			t.Fatalf("read scope alias = %v", err)
		}
	})

	t.Run("scope is file", func(t *testing.T) {
		root := t.TempDir()
		if err := os.WriteFile(filepath.Join(root, "team"), []byte("file"), 0o600); err != nil {
			t.Fatal(err)
		}
		source, _ := New(Config{Root: root})
		if _, err := source.List(context.Background(), "team", 1); err == nil {
			t.Fatal("listed a file scope")
		}
	})

	t.Run("manifest symlink escape", func(t *testing.T) {
		root := t.TempDir()
		directory := filepath.Join(root, "team", "skill")
		if err := os.MkdirAll(directory, 0o700); err != nil {
			t.Fatal(err)
		}
		outside := filepath.Join(t.TempDir(), "SKILL.md")
		if err := os.WriteFile(outside, []byte("---\nname: skill\ndescription: value\n---\nbody"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(outside, filepath.Join(directory, "SKILL.md")); err != nil {
			t.Fatal(err)
		}
		source, _ := New(Config{Root: root})
		if _, err := source.List(context.Background(), "team", 1); err == nil || !stringsContains(err.Error(), "escapes") {
			t.Fatalf("list manifest escape = %v", err)
		}
		if _, err := source.Read(context.Background(), "team", "skill", 100); err == nil || !stringsContains(err.Error(), "escapes") {
			t.Fatalf("read manifest escape = %v", err)
		}
	})

	t.Run("manifest read and parse errors", func(t *testing.T) {
		root := t.TempDir()
		malformed := writeSkill(t, root, "team", "malformed", "malformed", "value", "body")
		if err := os.WriteFile(filepath.Join(malformed, "SKILL.md"), []byte("no frontmatter"), 0o600); err != nil {
			t.Fatal(err)
		}
		source, _ := New(Config{Root: root})
		if _, err := source.List(context.Background(), "team", 10); err == nil {
			t.Fatal("listed malformed manifest")
		}

		root = t.TempDir()
		directory := filepath.Join(root, "team", "directory")
		if err := os.MkdirAll(filepath.Join(directory, "SKILL.md"), 0o700); err != nil {
			t.Fatal(err)
		}
		source, _ = New(Config{Root: root})
		if _, err := source.List(context.Background(), "team", 10); err == nil {
			t.Fatal("listed directory manifest")
		}
	})

	t.Run("bounded and mismatched reads", func(t *testing.T) {
		root := t.TempDir()
		writeSkill(t, root, "team", "large", "large", "value", strings.Repeat("x", 20<<10))
		writeSkill(t, root, "team", "wrong", "other", "value", "body")
		source, _ := New(Config{Root: root, MaxDescriptionBytes: 32})
		if _, err := source.Read(context.Background(), "team", "large", 1); !errors.Is(err, skillcore.ErrLimitExceeded) {
			t.Fatalf("manifest read bound = %v", err)
		}
		if _, err := source.Read(context.Background(), "team", "wrong", 100); err == nil || !stringsContains(err.Error(), "declares name") {
			t.Fatalf("mismatched read = %v", err)
		}
	})

	t.Run("skill symlink escape on read", func(t *testing.T) {
		root := t.TempDir()
		outside := t.TempDir()
		writeSkill(t, outside, ".", "evil", "evil", "value", "body")
		if err := os.MkdirAll(filepath.Join(root, "team"), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(filepath.Join(outside, "evil"), filepath.Join(root, "team", "evil")); err != nil {
			t.Fatal(err)
		}
		source, _ := New(Config{Root: root})
		if _, err := source.Read(context.Background(), "team", "evil", 100); err == nil || !stringsContains(err.Error(), "escapes") {
			t.Fatalf("skill read escape = %v", err)
		}
	})
}

func filepathConfig(root string) Config { return Config{Root: root} }

func stringsContains(value, fragment string) bool {
	for index := 0; index+len(fragment) <= len(value); index++ {
		if value[index:index+len(fragment)] == fragment {
			return true
		}
	}
	return false
}
