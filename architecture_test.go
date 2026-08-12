package agentic_test

import (
	"bufio"
	"bytes"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// The rules in this file are the ones ARCHITECTURE.md states. They are tests
// rather than prose because a layout described in a document and a layout
// enforced by the build diverge the first time someone is in a hurry, and the
// document is what loses. Each failure message names the decision it is
// protecting, so a legitimate change is a matter of editing both together.

// moduleRule is what ARCHITECTURE.md says about one module.
type moduleRule struct {
	// inWorkspace is whether go.work lists it. The two that are not need CGO
	// and a native ONNX Runtime, and listing them would make `go build ./...`
	// at the repository root fail for anyone who has not installed those.
	inWorkspace bool
	// replacesRoot is whether its go.mod redirects the root module to this
	// checkout. Release-consumer modules do not; repository acceptance modules
	// do — see the test below.
	replacesRoot bool
}

// modules is every Go module in the repository. A nested module exists to keep
// something out of the dependency graph of `go get
// github.com/regularkevvv/agentic`, and for no other reason.
var modules = map[string]moduleRule{
	".":                        {inWorkspace: true},
	"harness":                  {inWorkspace: true, replacesRoot: false},
	"harness/codemode/gomonty": {inWorkspace: true, replacesRoot: false},
	"harness/sessionloop":      {inWorkspace: true, replacesRoot: false},
	"harness/store/postgres":   {inWorkspace: true, replacesRoot: false},
	"tui":                      {inWorkspace: true, replacesRoot: false},
	"e2e":                      {inWorkspace: true, replacesRoot: true},
	"provider/local/onnx":      {inWorkspace: false, replacesRoot: false},
	"e2e/localinference":       {inWorkspace: false, replacesRoot: true},
}

// internalPackages is every package directly under internal/. The list is short
// on purpose: internal/ is where the two halves of the library live, and a
// fifth top-level entry means a third thing has appeared that is neither.
var internalPackages = map[string]string{
	"core":         "chat primitives",
	"retrieval":    "vector primitives",
	"providerhttp": "HTTP transport shared by providers",
	"testutil":     "test doubles for this repository's own tests",
}

// topLevelDirectories is the repository's first level, excluding dot-prefixed
// directories, which are tooling rather than code.
var topLevelDirectories = map[string]string{
	"docs":     "documentation",
	"e2e":      "nested module: live tests and examples",
	"harness":  "nested module: durable sessions",
	"internal": "the two halves, not importable by callers",
	"mcp":      "Model Context Protocol client",
	"provider": "every provider",
	"testdata": "compile-failure fixtures",
	"tool":     "tool builders and toolsets",
	"tui":      "nested module: reusable terminal client",
}

// capabilityInterfaces are the contracts a provider package can satisfy. A
// provider that satisfies none of them is not a provider.
var capabilityInterfaces = map[string]bool{
	"Model":                 true,
	"StreamModel":           true,
	"Embedder":              true,
	"RepresentationEncoder": true,
	"Reranker":              true,
}

func repoRoot(t *testing.T) string {
	t.Helper()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate repository root")
	}
	return filepath.Dir(filename)
}

// requireRepositoryCheckout skips repository-topology assertions when this
// package is being tested from a published module archive. Go module archives
// intentionally omit nested modules and VCS metadata, so those assertions can
// only describe a source checkout. Package-boundary tests still run in both
// environments.
func requireRepositoryCheckout(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(repoRoot(t), ".git")); os.IsNotExist(err) {
		t.Skip("repository topology is unavailable in a published module archive")
	} else if err != nil {
		t.Fatalf("inspect repository metadata: %v", err)
	}
}

// TestRepositoryHasExactlyTheDocumentedModules fails when a go.mod appears or
// disappears without ARCHITECTURE.md and the map above being updated with it.
//
// This is the check that makes the next one meaningful: without it, a new
// module could be left out of go.work and out of this file at once, and the
// workspace test below would happily pass over a module it had never heard of.
func TestRepositoryHasExactlyTheDocumentedModules(t *testing.T) {
	requireRepositoryCheckout(t)
	t.Parallel()
	root := repoRoot(t)

	found := make(map[string]bool)
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			// Dot-directories are tooling, and a vendor tree is someone else's
			// modules rather than this repository's.
			if path != root && (strings.HasPrefix(entry.Name(), ".") || entry.Name() == "vendor") {
				return fs.SkipDir
			}
			return nil
		}
		if entry.Name() != "go.mod" {
			return nil
		}
		relative, err := filepath.Rel(root, filepath.Dir(path))
		if err != nil {
			return err
		}
		found[filepath.ToSlash(relative)] = true
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	for module := range found {
		if _, documented := modules[module]; !documented {
			t.Errorf("module %q exists but is not in ARCHITECTURE.md or the modules map here.\n"+
				"A nested module keeps something out of `go get`'s dependency graph. Say what, "+
				"in both places, and decide whether go.work should list it.", module)
		}
	}
	for module := range modules {
		if !found[module] {
			t.Errorf("module %q is documented here but has no go.mod", module)
		}
	}
}

// TestWorkspaceListsEveryNonCGOModule keeps go.work honest in both directions.
//
// Forgetting to add a module makes it invisible to every command run from the
// repository root, which is how a module rots. Adding a CGO module breaks
// `go build ./...` for every contributor who has not installed a native ONNX
// Runtime, which is how a repository becomes unpleasant to check out. Both are
// silent, and both are one line.
func TestWorkspaceListsEveryNonCGOModule(t *testing.T) {
	requireRepositoryCheckout(t)
	t.Parallel()
	root := repoRoot(t)

	file, err := os.Open(filepath.Join(root, "go.work"))
	if err != nil {
		t.Fatalf("open go.work: %v", err)
	}
	defer func() { _ = file.Close() }()

	listed := make(map[string]bool)
	inUseBlock := false
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if comment := strings.Index(line, "//"); comment >= 0 {
			line = strings.TrimSpace(line[:comment])
		}
		switch {
		case line == "":
		case strings.HasPrefix(line, "use ("):
			inUseBlock = true
		case inUseBlock && line == ")":
			inUseBlock = false
		case inUseBlock:
			listed[normalizeModulePath(line)] = true
		case strings.HasPrefix(line, "use "):
			listed[normalizeModulePath(strings.TrimPrefix(line, "use "))] = true
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("read go.work: %v", err)
	}

	for module, rule := range modules {
		shouldBeListed := rule.inWorkspace
		switch {
		case shouldBeListed && !listed[module]:
			t.Errorf("go.work does not list %q. Every module that builds without a native "+
				"toolchain belongs in the workspace, or the commands run from the repository "+
				"root will never see it.", module)
		case !shouldBeListed && listed[module]:
			t.Errorf("go.work lists %q, which needs CGO and a native ONNX Runtime. Adding it "+
				"here makes `go build ./...` fail for everyone who has not installed those. "+
				"It is gated by the `onnx` CI job instead.", module)
		}
	}
	for module := range listed {
		if _, known := modules[module]; !known {
			t.Errorf("go.work lists %q, which is not a module this repository documents", module)
		}
	}
}

// TestNestedModulesResolveTheRootAsDocumented protects a distinction that is
// one line from being erased and leaves no trace when it is.
//
// Harness, the optional GoMonty adapter, TUI, and the native ONNX provider
// deliberately have no `replace`: their ordered release checks rehearse what
// users obtain through `go get`. go.work still supplies the non-CGO modules
// during a coordinated change. Adding a replace would make that consumer
// distinction disappear silently.
//
// The others replace the root because they are not consumers, they are parts of
// this repository that happen to need their own dependency graph.
func TestNestedModulesResolveTheRootAsDocumented(t *testing.T) {
	requireRepositoryCheckout(t)
	t.Parallel()
	root := repoRoot(t)

	for module, rule := range modules {
		if module == "." {
			continue
		}
		contents, err := os.ReadFile(filepath.Join(root, module, "go.mod"))
		if err != nil {
			t.Errorf("read %s/go.mod: %v", module, err)
			continue
		}
		replaces := false
		for _, line := range strings.Split(string(contents), "\n") {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "replace github.com/regularkevvv/agentic ") ||
				strings.HasPrefix(line, "replace github.com/regularkevvv/agentic=") {
				replaces = true
			}
		}
		switch {
		case rule.replacesRoot && !replaces:
			t.Errorf("%s/go.mod does not replace the root module. It would then build against "+
				"a published version rather than this checkout, so a change here would not "+
				"reach it until after a release.", module)
		case !rule.replacesRoot && replaces:
			t.Errorf("%s/go.mod replaces the root module, but it is documented as a release consumer. "+
				"An ordered GOWORK=off release check must exercise published dependencies; "+
				"a replace directive makes it test this checkout instead. See ARCHITECTURE.md.", module)
		}
	}
}

// TestSessionloopModuleHasNoProjectDependencies keeps the session protocol
// module at zero dependencies, which is the property the module exists for.
//
// harness/sessionloop is the provider-neutral protocol that Harness, the TUI
// bridge, and external facades all import; the moment its go.mod gains a
// require or a replace, every one of those consumers inherits that graph and
// the neutral boundary is gone. Plan section 5.3 makes this explicit: adding a
// requirement to harness/sessionloop/go.mod demands an explicit architectural
// revision — edit ARCHITECTURE.md, the plan, and this test together, or not at
// all.
func TestSessionloopModuleHasNoProjectDependencies(t *testing.T) {
	requireRepositoryCheckout(t)
	t.Parallel()
	root := repoRoot(t)

	contents, err := os.ReadFile(filepath.Join(root, "harness", "sessionloop", "go.mod"))
	if err != nil {
		t.Fatalf("read harness/sessionloop/go.mod: %v", err)
	}
	for _, line := range strings.Split(string(contents), "\n") {
		line = strings.TrimSpace(line)
		if comment := strings.Index(line, "//"); comment >= 0 {
			line = strings.TrimSpace(line[:comment])
		}
		if strings.HasPrefix(line, "require") || strings.HasPrefix(line, "replace") ||
			strings.Contains(line, " require ") || strings.Contains(line, " replace ") {
			t.Errorf("harness/sessionloop/go.mod contains %q.\n"+
				"The session protocol module has zero require and replace directives by design: "+
				"it is what keeps the protocol importable without Agentic, Harness, the TUI, or a "+
				"provider SDK entering a consumer's module graph. Plan section 5.3 requires an "+
				"explicit architectural revision before any dependency is added — update "+
				"ARCHITECTURE.md and this test in the same change, or remove the directive.", line)
		}
	}
}

// TestReleaseOrderDocumentsSessionloopFirst freezes the documented tag order.
//
// harness/go.mod requires a published sessionloop tag and carries no replace,
// so releasing Harness before sessionloop deadlocks the GOWORK=off release
// check (plan section 14). Remote tags are not hermetically testable; the
// documented order in ARCHITECTURE.md is what release engineering follows,
// so this test keeps that sentence from silently regressing.
func TestReleaseOrderDocumentsSessionloopFirst(t *testing.T) {
	requireRepositoryCheckout(t)
	t.Parallel()
	root := repoRoot(t)

	contents, err := os.ReadFile(filepath.Join(root, "ARCHITECTURE.md"))
	if err != nil {
		t.Fatalf("read ARCHITECTURE.md: %v", err)
	}
	// Markdown wraps sentences across lines, and the wrap point is not part
	// of what this test protects.
	text := strings.Join(strings.Fields(string(contents)), " ")

	marker := "released in dependency order:"
	start := strings.Index(text, marker)
	if start < 0 {
		t.Fatalf("ARCHITECTURE.md no longer contains %q; the release-order sentence is the "+
			"only place the tag order is written down, so it must exist and this test must "+
			"point at it", marker)
	}
	sentence := text[start:]
	if end := strings.Index(sentence, "."); end >= 0 {
		sentence = sentence[:end]
	}
	sessionloop := strings.Index(sentence, "sessionloop")
	harnessAt := strings.Index(sentence, "Harness")
	switch {
	case sessionloop < 0:
		t.Errorf("ARCHITECTURE.md's release-order sentence does not mention sessionloop. "+
			"Harness requires a published sessionloop tag, so sessionloop must be tagged "+
			"before Harness and the documented order must say so. Sentence: %q", sentence)
	case harnessAt < 0:
		t.Errorf("ARCHITECTURE.md's release-order sentence does not mention Harness: %q", sentence)
	case sessionloop > harnessAt:
		t.Errorf("ARCHITECTURE.md's release-order sentence lists Harness before sessionloop. "+
			"harness/go.mod requires a published sessionloop tag with no replace directive, so "+
			"tagging Harness first deadlocks its GOWORK=off release check. Sentence: %q", sentence)
	}
}

// normalizeModulePath turns a go.work entry ("./harness", ".") into the form
// the modules map uses ("harness", ".").
func normalizeModulePath(entry string) string {
	entry = strings.Trim(strings.TrimSpace(entry), `"`)
	if entry == "." {
		return "."
	}
	return strings.TrimSuffix(strings.TrimPrefix(entry, "./"), "/")
}

// TestChatAndRetrievalHalvesDoNotReferenceEachOther is the boundary this
// library is organized around.
//
// Chat and retrieval share providers and nothing else: a program that only
// embeds text never constructs an agent. The halves erode by convenience — one
// field on a request struct, one shared error type — and after that neither can
// be read, tested, or moved without the other. Test files are included because
// a test that reaches across is the first draft of code that does.
func TestChatAndRetrievalHalvesDoNotReferenceEachOther(t *testing.T) {
	t.Parallel()
	root := repoRoot(t)

	const modulePath = "github.com/regularkevvv/agentic/"
	halves := []struct{ dir, forbidden string }{
		{"internal/core", modulePath + "internal/retrieval"},
		{"internal/retrieval", modulePath + "internal/core"},
	}
	for _, half := range halves {
		for file, imports := range importsUnder(t, filepath.Join(root, half.dir)) {
			for _, imported := range imports {
				if strings.HasPrefix(imported, half.forbidden) {
					relative, _ := filepath.Rel(root, file)
					t.Errorf("%s imports %q.\n"+
						"internal/core and internal/retrieval are the two halves of this library "+
						"and do not reference each other; see ARCHITECTURE.md. If they genuinely "+
						"need to share something, it belongs in a third package that both import.",
						filepath.ToSlash(relative), imported)
				}
			}
		}
	}
}

// TestInternalHoldsOnlyDocumentedPackages guards the shape of internal/ rather
// than its contents. Four entries is not a target; it is the count that follows
// from there being two halves, one thing they share, and one test helper.
func TestInternalHoldsOnlyDocumentedPackages(t *testing.T) {
	t.Parallel()
	assertDirectoryMatches(t, filepath.Join(repoRoot(t), "internal"), internalPackages,
		"internal/%s exists but ARCHITECTURE.md does not describe it. It is either part of "+
			"the chat half, part of the retrieval half, or something both need — say which.")
}

// TestTopLevelDirectoriesAreDocumented catches the directory that appears
// because something did not obviously belong anywhere, which is the moment a
// layout starts costing people time.
func TestTopLevelDirectoriesAreDocumented(t *testing.T) {
	requireRepositoryCheckout(t)
	t.Parallel()
	assertDirectoryMatches(t, repoRoot(t), topLevelDirectories,
		"%s/ is a new top-level directory and ARCHITECTURE.md does not name it. "+
			"Add it to the tree there and to this test, or find it a home in an existing one.")
}

// assertDirectoryMatches compares the sub-directories of dir against the
// documented set, ignoring dot-prefixed entries.
func assertDirectoryMatches(t *testing.T, dir string, documented map[string]string, unexpected string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read %s: %v", dir, err)
	}
	present := make(map[string]bool)
	for _, entry := range entries {
		if !entry.IsDir() || strings.HasPrefix(entry.Name(), ".") {
			continue
		}
		present[entry.Name()] = true
		if _, ok := documented[entry.Name()]; !ok {
			t.Errorf(unexpected, entry.Name())
		}
	}
	missing := make([]string, 0)
	for name := range documented {
		if !present[name] {
			missing = append(missing, name)
		}
	}
	sort.Strings(missing)
	for _, name := range missing {
		t.Errorf("%s is documented as %q but does not exist", name, documented[name])
	}
}

// executableMagic are the leading bytes of the binary formats a Go build can
// produce, plus the static archive a cgo module links against.
var executableMagic = [][]byte{
	{0x7f, 'E', 'L', 'F'},          // ELF: Linux
	{0xfe, 0xed, 0xfa, 0xce},       // Mach-O 32-bit
	{0xfe, 0xed, 0xfa, 0xcf},       // Mach-O 64-bit
	{0xce, 0xfa, 0xed, 0xfe},       // Mach-O 32-bit, byte-swapped
	{0xcf, 0xfa, 0xed, 0xfe},       // Mach-O 64-bit, byte-swapped
	{0xca, 0xfe, 0xba, 0xbe},       // Mach-O universal
	{'M', 'Z'},                     // PE: Windows
	{'!', '<', 'a', 'r', 'c', 'h'}, // ar archive: libtokenizers.a and friends
}

// TestNoCompiledBinariesAreTracked fails if a build artifact has been committed.
//
// `go build ./...` writes one executable per main package into the directory it
// runs from, named after the package and with no extension, so the *.exe and
// *.dylib rules in .gitignore do not match it on macOS or Linux. A 26 MiB
// Mach-O binary reached main that way, from a routine `go build ./...` in an
// example module followed by `git add -A`.
//
// .gitignore now names the binaries the current examples produce, but that list
// goes stale the moment someone adds an example. This does not: it asks git what
// is tracked and reads the first bytes of each file.
func TestNoCompiledBinariesAreTracked(t *testing.T) {
	t.Parallel()
	root := repoRoot(t)

	command := exec.Command("git", "ls-files", "-z")
	command.Dir = root
	output, err := command.Output()
	if err != nil {
		t.Skipf("git ls-files unavailable, cannot check tracked files: %v", err)
	}

	longest := 0
	for _, magic := range executableMagic {
		if len(magic) > longest {
			longest = len(magic)
		}
	}

	for _, name := range strings.Split(string(output), "\x00") {
		if name == "" {
			continue
		}
		file, err := os.Open(filepath.Join(root, name))
		if err != nil {
			// A tracked file missing from the worktree is a different problem,
			// and one git itself reports.
			continue
		}
		header := make([]byte, longest)
		read, _ := io.ReadFull(file, header)
		_ = file.Close()

		for _, magic := range executableMagic {
			if read >= len(magic) && bytes.Equal(header[:len(magic)], magic) {
				t.Errorf("%s is a compiled binary and is tracked in git.\n"+
					"Build artifacts do not belong in a library repository: they are "+
					"platform-specific, they are large, and they stay in history after "+
					"deletion. Remove it, and add its name to .gitignore.", name)
				break
			}
		}
	}
}

// TestMarkdownLinksResolve checks that every relative link in every Markdown
// file points at something that exists.
//
// Moving a document is the cheapest possible way to make a set of documents
// wrong, and the result is invisible: the repository builds, the tests pass, and
// a reader following the link gets a 404 from GitHub. This found two dead links
// the first time it ran — one left by a module that moved a directory deeper,
// and one by the docs/design split.
//
// Only the path is checked, not the anchor. Verifying anchors means parsing
// headings and their slug rules, which is more machinery than the problem
// deserves.
func TestMarkdownLinksResolve(t *testing.T) {
	requireRepositoryCheckout(t)
	t.Parallel()
	root := repoRoot(t)

	// Code blocks and inline spans are stripped first: Go generics in a fenced
	// example read as Markdown links — RequireDriver[string](agent) is a
	// perfect one — and a checker that cries wolf gets switched off.
	fences := regexp.MustCompile("(?s)```.*?```")
	spans := regexp.MustCompile("`[^`\n]*`")
	links := regexp.MustCompile(`\[[^\]]*\]\(([^)\s]+)\)`)

	checked := 0
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			if path != root && strings.HasPrefix(entry.Name(), ".") {
				return fs.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".md" {
			return nil
		}
		contents, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		text := spans.ReplaceAllString(fences.ReplaceAllString(string(contents), ""), "")
		for _, match := range links.FindAllStringSubmatch(text, -1) {
			target := match[1]
			if strings.Contains(target, "://") || strings.HasPrefix(target, "mailto:") ||
				strings.HasPrefix(target, "#") {
				continue
			}
			if anchor := strings.IndexByte(target, '#'); anchor >= 0 {
				target = target[:anchor]
			}
			if target == "" {
				continue
			}
			checked++
			if _, err := os.Stat(filepath.Join(filepath.Dir(path), target)); err != nil {
				relative, _ := filepath.Rel(root, path)
				t.Errorf("%s links to %q, which does not exist",
					filepath.ToSlash(relative), match[1])
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if checked < 20 {
		t.Fatalf("checked only %d relative links, expected many more", checked)
	}
}

// TestEveryProviderDeclaresItsCapabilities enforces the one thing the directory
// tree cannot say.
//
// provider/ is flat and most of its packages hold several roles at once —
// bedrock is a model and an embedder, cohere an embedder and a reranker — so no
// arrangement of directories can encode capability, and grouping by role would
// have to put those packages in two places. The compile-time assertion is
// therefore the only machine-readable answer to "what is this provider for",
// and this test is what stops it being optional. A new provider that forgets it
// fails here rather than being discovered by reading its source.
func TestEveryProviderDeclaresItsCapabilities(t *testing.T) {
	t.Parallel()
	providerRoot := filepath.Join(repoRoot(t), "provider")

	entries, err := os.ReadDir(providerRoot)
	if err != nil {
		t.Fatalf("read provider directory: %v", err)
	}

	checked := 0
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		dir := filepath.Join(providerRoot, entry.Name())

		// A package with its own go.mod is a separate module and is compiled,
		// tested, and asserted by itself. Walking into it from here would test
		// source this module never builds.
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			continue
		}
		// provider/test holds the test doubles and the conformance suite rather
		// than a provider, and is exercised by the packages that consume it.
		if entry.Name() == "test" {
			continue
		}
		// A directory that groups providers rather than being one — provider/local
		// holds only the nested onnx module — has no Go files of its own and
		// nothing to assert.
		if !hasGoFiles(dir) {
			continue
		}

		if len(declaredCapabilities(t, dir)) == 0 {
			t.Errorf("provider/%s declares no capability: add a compile-time "+
				"assertion such as `var _ retrieval.Embedder = (*Embedder)(nil)` so "+
				"what this package implements is checkable rather than folklore",
				entry.Name())
			continue
		}
		checked++
	}

	// A refactor that moved or renamed provider/ would otherwise leave this
	// test passing over nothing.
	if checked < 10 {
		t.Fatalf("checked only %d provider packages, expected the full set", checked)
	}
}

// hasGoFiles reports whether dir is itself a package rather than a directory
// that only groups others.
func hasGoFiles(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, entry := range entries {
		name := entry.Name()
		if !entry.IsDir() && filepath.Ext(name) == ".go" && !strings.HasSuffix(name, "_test.go") {
			return true
		}
	}
	return false
}

// declaredCapabilities returns the capability interfaces a package asserts,
// reading `var _ retrieval.Embedder = …` declarations from its non-test files.
func declaredCapabilities(t *testing.T, dir string) map[string]bool {
	t.Helper()
	found := make(map[string]bool)
	err := filepath.WalkDir(dir, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			if path == dir {
				return nil
			}
			if _, err := os.Stat(filepath.Join(path, "go.mod")); err == nil {
				return fs.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".go" || strings.HasSuffix(path, "_test.go") {
			return nil
		}

		file, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			return err
		}
		for _, decl := range file.Decls {
			genDecl, ok := decl.(*ast.GenDecl)
			if !ok || genDecl.Tok != token.VAR {
				continue
			}
			for _, spec := range genDecl.Specs {
				value, ok := spec.(*ast.ValueSpec)
				if !ok || value.Type == nil {
					continue
				}
				// Only the blank identifier: `var _ retrieval.Embedder = …` is an
				// assertion, whereas a named variable of that type is a field.
				if len(value.Names) != 1 || value.Names[0].Name != "_" {
					continue
				}
				selector, ok := value.Type.(*ast.SelectorExpr)
				if !ok {
					continue
				}
				if capabilityInterfaces[selector.Sel.Name] {
					found[selector.Sel.Name] = true
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("%s: %v", dir, err)
	}
	return found
}

// importsUnder returns the import paths of every Go file below dir, keyed by
// file. Imports are read with the parser rather than `go list` so that files
// excluded by a build tag are still checked: a rule that a build constraint can
// switch off is not a rule.
func importsUnder(t *testing.T, dir string) map[string][]string {
	t.Helper()
	result := make(map[string][]string)
	err := filepath.WalkDir(dir, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil || entry.IsDir() || filepath.Ext(path) != ".go" {
			return walkErr
		}
		file, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		ast.Inspect(file, func(node ast.Node) bool {
			spec, ok := node.(*ast.ImportSpec)
			if !ok {
				return true
			}
			if value, unquoteErr := strconv.Unquote(spec.Path.Value); unquoteErr == nil {
				result[path] = append(result[path], value)
			}
			return false
		})
		return nil
	})
	if err != nil {
		t.Fatalf("read imports under %s: %v", dir, err)
	}
	return result
}
