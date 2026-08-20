//go:build linux && (amd64 || arm64)

package sandbox

import (
	"errors"
	"slices"
	"strings"
	"testing"

	seccomp "github.com/elastic/go-seccomp-bpf"
	"github.com/landlock-lsm/go-landlock/landlock"
)

func TestLinuxBootstrap(t *testing.T) {
	arguments := []string{"helper", bootstrapArgument, "root", "temporary", "cwd", "target", "network", "arg"}
	environment := []string{"KEEP=value", bootstrapEnv + "=1"}
	var gotRules int
	var networkAllowed bool
	var changedDirectory string
	var execTarget string
	var execArguments, execEnvironment []string
	err := linuxBootstrap(
		arguments,
		environment,
		func() (int, error) { return 3, nil },
		func(_ landlock.Config, rules ...landlock.Rule) error {
			gotRules = len(rules)
			return nil
		},
		func(filter seccomp.Filter) error {
			networkAllowed = !slices.Contains(filter.Policy.Syscalls[0].Names, "socket")
			return nil
		},
		func(path string) error { changedDirectory = path; return nil },
		func(target string, arguments, environment []string) error {
			execTarget, execArguments, execEnvironment = target, arguments, environment
			return nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if gotRules != 20 || !networkAllowed {
		t.Fatalf("rule count = %d, network allowed = %v", gotRules, networkAllowed)
	}
	if changedDirectory != "cwd" || execTarget != "target" {
		t.Fatalf("chdir = %q, exec target = %q", changedDirectory, execTarget)
	}
	if !slices.Equal(execArguments, []string{"target", "arg"}) || !slices.Equal(execEnvironment, []string{"KEEP=value"}) {
		t.Fatalf("exec arguments = %#v, environment = %#v", execArguments, execEnvironment)
	}

	enterFailure := errors.New("enter failure")
	if err := linuxBootstrap(arguments, environment,
		func() (int, error) { return 0, enterFailure },
		func(landlock.Config, ...landlock.Rule) error {
			t.Fatal("restrict called after ABI failure")
			return nil
		},
		func(seccomp.Filter) error { t.Fatal("seccomp called after ABI failure"); return nil },
		func(string) error { t.Fatal("chdir called after enter failure"); return nil },
		func(string, []string, []string) error { t.Fatal("exec called after enter failure"); return nil },
	); !errors.Is(err, enterFailure) {
		t.Fatalf("enter failure = %v", err)
	}

	chdirFailure := errors.New("chdir failure")
	if err := linuxBootstrap(arguments, environment,
		func() (int, error) { return 3, nil },
		func(landlock.Config, ...landlock.Rule) error { return nil },
		func(seccomp.Filter) error { return nil },
		func(string) error { return chdirFailure },
		func(string, []string, []string) error { t.Fatal("exec called after chdir failure"); return nil },
	); !errors.Is(err, chdirFailure) || !strings.Contains(err.Error(), "chdir") {
		t.Fatalf("chdir failure = %v", err)
	}

	execFailure := errors.New("exec failure")
	if err := linuxBootstrap(arguments, environment,
		func() (int, error) { return 3, nil },
		func(landlock.Config, ...landlock.Rule) error { return nil },
		func(seccomp.Filter) error { return nil },
		func(string) error { return nil },
		func(string, []string, []string) error { return execFailure },
	); !errors.Is(err, execFailure) || !strings.Contains(err.Error(), "exec") {
		t.Fatalf("exec failure = %v", err)
	}
}

func TestLinuxBootstrapMain(t *testing.T) {
	arguments := []string{"helper", bootstrapArgument, "root", "temporary", "cwd", "target", "no-network"}
	var output strings.Builder
	exitCode := 0
	linuxBootstrapMain(
		arguments,
		nil,
		&output,
		func() (int, error) { return 0, errors.New("blocked") },
		func(landlock.Config, ...landlock.Rule) error { return nil },
		func(seccomp.Filter) error { return nil },
		func(string) error { return nil },
		func(string, []string, []string) error { return nil },
		func(code int) { exitCode = code },
	)
	if exitCode != 126 || !strings.Contains(output.String(), "agentic sandbox: blocked") {
		t.Fatalf("exit = %d, stderr = %q", exitCode, output.String())
	}

	linuxBootstrapMain(
		arguments,
		nil,
		&output,
		func() (int, error) { return 3, nil },
		func(landlock.Config, ...landlock.Rule) error { return nil },
		func(seccomp.Filter) error { return nil },
		func(string) error { return nil },
		func(string, []string, []string) error { return nil },
		func(int) { t.Fatal("successful bootstrap exited") },
	)
}

func TestProbeLinuxBackend(t *testing.T) {
	probeFailure := errors.New("probe failure")
	if _, err := probeLinuxBackend(func() (int, error) { return 0, probeFailure }); !errors.Is(err, probeFailure) {
		t.Fatalf("probe failure = %v", err)
	}
	if _, err := probeLinuxBackend(func() (int, error) { return 0, nil }); err == nil {
		t.Fatal("missing Landlock ABI succeeded")
	}
	if backend, err := probeLinuxBackend(func() (int, error) { return 1, nil }); err != nil || backend != "landlock+seccomp" {
		t.Fatalf("backend = %q, %v", backend, err)
	}
}

func TestConfigureLinuxSandbox(t *testing.T) {
	abiFailure := errors.New("ABI failure")
	if err := configureLinuxSandbox("root", "temporary", "target", false,
		func() (int, error) { return 0, abiFailure },
		func(landlock.Config, ...landlock.Rule) error {
			t.Fatal("restriction called after ABI failure")
			return nil
		},
		func(seccomp.Filter) error {
			t.Fatal("seccomp called after ABI failure")
			return nil
		},
	); !errors.Is(err, abiFailure) {
		t.Fatalf("ABI failure = %v", err)
	}

	restrictFailure := errors.New("restriction failure")
	if err := configureLinuxSandbox("root", "temporary", "target", false,
		func() (int, error) { return 3, nil },
		func(landlock.Config, ...landlock.Rule) error { return restrictFailure },
		func(seccomp.Filter) error {
			t.Fatal("seccomp called after restriction failure")
			return nil
		},
	); !errors.Is(err, restrictFailure) || !strings.Contains(err.Error(), "restrict with landlock") {
		t.Fatalf("restriction failure = %v", err)
	}

	installFailure := errors.New("seccomp failure")
	var gotRules int
	if err := configureLinuxSandbox("root", "temporary", "target", true,
		func() (int, error) { return 3, nil },
		func(_ landlock.Config, rules ...landlock.Rule) error {
			gotRules = len(rules)
			return nil
		},
		func(filter seccomp.Filter) error {
			if slices.Contains(filter.Policy.Syscalls[0].Names, "socket") {
				t.Fatal("network policy was not forwarded")
			}
			return installFailure
		},
	); !errors.Is(err, installFailure) || gotRules != 20 {
		t.Fatalf("seccomp failure = %v, rule count = %d", err, gotRules)
	}
}

func TestLinuxLandlockPolicySelectsKernelABI(t *testing.T) {
	for _, test := range []struct {
		version int
		want    string
	}{
		{version: 1, want: "Landlock V1"},
		{version: 2, want: "Landlock V2"},
		{version: 3, want: "Landlock V3"},
		{version: 9, want: "Landlock V3"},
	} {
		config, rules := linuxLandlockPolicy(test.version, "/workspace", "/temporary", "/usr/bin/tool")
		if got := config.String(); !strings.Contains(got, test.want) {
			t.Fatalf("ABI %d config = %q, want %q", test.version, got, test.want)
		}
		if len(rules) != 20 {
			t.Fatalf("ABI %d rule count = %d, want 20", test.version, len(rules))
		}
	}
}

func TestLinuxSeccompFilterTracksNetworkPolicy(t *testing.T) {
	for _, test := range []struct {
		network    bool
		wantSocket bool
	}{
		{network: false, wantSocket: true},
		{network: true, wantSocket: false},
	} {
		filter := linuxSeccompFilter(test.network)
		if !filter.NoNewPrivs || filter.Flag != seccomp.FilterFlagTSync {
			t.Fatalf("network=%v filter hardening = %#v", test.network, filter)
		}
		if filter.Policy.DefaultAction != seccomp.ActionAllow || len(filter.Policy.Syscalls) != 1 {
			t.Fatalf("network=%v policy = %#v", test.network, filter.Policy)
		}
		group := filter.Policy.Syscalls[0]
		if group.Action != seccomp.ActionErrno {
			t.Fatalf("network=%v deny action = %v", test.network, group.Action)
		}
		if got := slices.Contains(group.Names, "socket"); got != test.wantSocket {
			t.Fatalf("network=%v socket denial = %v, want %v", test.network, got, test.wantSocket)
		}
		if got := slices.Contains(group.Names, "mount"); !got {
			t.Fatalf("network=%v mount denial is absent", test.network)
		}
	}
}

func TestLoadSeccompFilter(t *testing.T) {
	loadFailure := errors.New("load failure")
	if err := loadSeccompFilter(false, func(filter seccomp.Filter) error {
		if !slices.Contains(filter.Policy.Syscalls[0].Names, "socket") {
			t.Fatal("network denial was not passed to loader")
		}
		return loadFailure
	}); !errors.Is(err, loadFailure) || !strings.Contains(err.Error(), "install seccomp filter") {
		t.Fatalf("load failure = %v", err)
	}
	if err := loadSeccompFilter(true, func(seccomp.Filter) error { return nil }); err != nil {
		t.Fatalf("successful load = %v", err)
	}
}

func TestWithoutEnvironment(t *testing.T) {
	got := withoutEnvironment([]string{"KEEP=one", "REMOVE=two", "MALFORMED", "KEEP=three"}, "REMOVE")
	want := []string{"KEEP=one", "MALFORMED", "KEEP=three"}
	if !slices.Equal(got, want) {
		t.Fatalf("environment = %#v, want %#v", got, want)
	}
}
