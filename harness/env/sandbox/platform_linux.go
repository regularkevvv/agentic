//go:build linux && (amd64 || arm64)

package sandbox

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"

	seccomp "github.com/elastic/go-seccomp-bpf"
	"github.com/landlock-lsm/go-landlock/landlock"
	landlocksyscall "github.com/landlock-lsm/go-landlock/landlock/syscall"
	"golang.org/x/sys/unix"

	harnessenv "github.com/regularkevvv/agentic/harness/env"
)

const (
	bootstrapArgument = "--agentic-sandbox-bootstrap"
	bootstrapEnv      = "AGENTIC_SANDBOX_BOOTSTRAP"
)

func init() {
	if os.Getenv(bootstrapEnv) != "1" || len(os.Args) < 7 || os.Args[1] != bootstrapArgument {
		return
	}
	linuxBootstrapMain(
		os.Args, os.Environ(), os.Stderr,
		landlockABI, landlock.Config.RestrictPaths, seccomp.LoadFilter,
		os.Chdir, unix.Exec, os.Exit,
	)
}

func linuxBootstrapMain(
	arguments, environment []string,
	stderr io.Writer,
	abi func() (int, error),
	restrict func(landlock.Config, ...landlock.Rule) error,
	load func(seccomp.Filter) error,
	chdir func(string) error,
	exec func(string, []string, []string) error,
	exit func(int),
) {
	if err := linuxBootstrap(arguments, environment, abi, restrict, load, chdir, exec); err != nil {
		_, _ = fmt.Fprintln(stderr, "agentic sandbox:", err)
		exit(126)
	}
}

func linuxBootstrap(
	arguments, environment []string,
	abi func() (int, error),
	restrict func(landlock.Config, ...landlock.Rule) error,
	load func(seccomp.Filter) error,
	chdir func(string) error,
	exec func(string, []string, []string) error,
) error {
	root, temporary, cwd, target := arguments[2], arguments[3], arguments[4], arguments[5]
	network := arguments[6] == "network"
	if err := configureLinuxSandbox(root, temporary, target, network, abi, restrict, load); err != nil {
		return err
	}
	if err := chdir(cwd); err != nil {
		return fmt.Errorf("chdir: %w", err)
	}
	environment = withoutEnvironment(environment, bootstrapEnv)
	commandArguments := append([]string{target}, arguments[7:]...)
	if err := exec(target, commandArguments, environment); err != nil {
		return fmt.Errorf("exec: %w", err)
	}
	return nil
}

func probeBackend() (string, error) {
	return probeLinuxBackend(landlockABI)
}

func probeLinuxBackend(abi func() (int, error)) (string, error) {
	version, err := abi()
	if err != nil {
		return "", fmt.Errorf("landlock unavailable: %w", err)
	}
	if version < 1 {
		return "", errors.New("landlock ABI is unavailable")
	}
	return "landlock+seccomp", nil
}

func execute(ctx context.Context, value execution) (harnessenv.CommandResult, error) {
	temporary, err := privateTemp()
	if err != nil {
		return harnessenv.CommandResult{}, err
	}
	defer func() { _ = os.RemoveAll(temporary) }()
	target, err := resolveCommand(value.command.Name, value.cwd)
	if err != nil {
		return harnessenv.CommandResult{}, err
	}
	helper, err := os.Executable()
	if err != nil {
		return harnessenv.CommandResult{}, err
	}
	network := "no-network"
	if value.network {
		network = "network"
	}
	forced := map[string]string{
		bootstrapEnv: "1",
		"TMPDIR":     temporary, "TMP": temporary, "TEMP": temporary,
		"GOCACHE": temporary + "/go-cache", "XDG_CACHE_HOME": temporary + "/cache",
	}
	environment := commandEnvironment(value.command.Env, forced)
	args := []string{bootstrapArgument, value.root, temporary, value.cwd, target, network}
	args = append(args, value.command.Args...)
	return run(ctx, helper, args, environment, value.cwd, value.command.Stdin)
}

func configureLinuxSandbox(
	root, temporary, target string,
	network bool,
	abi func() (int, error),
	restrict func(landlock.Config, ...landlock.Rule) error,
	load func(seccomp.Filter) error,
) error {
	version, err := abi()
	if err != nil {
		return err
	}
	config, rules := linuxLandlockPolicy(version, root, temporary, target)
	if err := restrict(config, rules...); err != nil {
		return fmt.Errorf("restrict with landlock: %w", err)
	}
	return loadSeccompFilter(network, load)
}

func linuxLandlockPolicy(version int, root, temporary, target string) (landlock.Config, []landlock.Rule) {
	readOnly := landlock.AccessFSSet(landlocksyscall.AccessFSReadFile | landlocksyscall.AccessFSReadDir)
	executable := readOnly | landlock.AccessFSSet(landlocksyscall.AccessFSExecute)
	rules := make([]landlock.Rule, 0, 20)
	for _, path := range []string{"/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/proc/self", "/proc/thread-self", "/opt", "/nix/store"} {
		rules = append(rules, landlock.PathAccess(readOnly, path).IgnoreIfMissing())
	}
	// The kernel treats an ELF interpreter as part of execution. Restrict that
	// grant to immutable system library trees rather than all system binaries.
	for _, path := range []string{"/lib", "/lib64", "/usr/lib"} {
		rules = append(rules, landlock.PathAccess(executable, path).IgnoreIfMissing())
	}
	helpers := commandHelpers(target)
	rules = append(rules,
		landlock.PathAccess(executable, helpers),
		landlock.PathAccess(
			landlock.AccessFSSet(landlocksyscall.AccessFSReadFile|landlocksyscall.AccessFSExecute),
			target,
		),
	)
	readWrite := landlock.RWDirs(root, temporary)
	var config landlock.Config
	switch {
	case version >= 3:
		config = landlock.V3
		readWrite = readWrite.WithRefer()
	case version >= 2:
		config = landlock.V2
		readWrite = readWrite.WithRefer()
	default:
		config = landlock.V1
	}
	rules = append(rules, readWrite)
	deviceAccess := landlock.AccessFSSet(landlocksyscall.AccessFSReadFile | landlocksyscall.AccessFSWriteFile)
	for _, path := range []string{"/dev/null", "/dev/tty", "/dev/random", "/dev/urandom"} {
		rules = append(rules, landlock.PathAccess(deviceAccess, path).IgnoreIfMissing())
	}
	return config, rules
}

func landlockABI() (int, error) {
	return landlocksyscall.LandlockGetABIVersion()
}

func loadSeccompFilter(network bool, load func(seccomp.Filter) error) error {
	if err := load(linuxSeccompFilter(network)); err != nil {
		return fmt.Errorf("install seccomp filter: %w", err)
	}
	return nil
}

func linuxSeccompFilter(network bool) seccomp.Filter {
	denied := []string{
		"ptrace", "mount", "umount2", "pivot_root", "chroot", "unshare", "setns", "bpf", "perf_event_open",
	}
	if !network {
		denied = append(denied, "socket", "socketpair")
	}
	return seccomp.Filter{
		NoNewPrivs: true,
		Flag:       seccomp.FilterFlagTSync,
		Policy: seccomp.Policy{
			DefaultAction: seccomp.ActionAllow,
			Syscalls: []seccomp.SyscallGroup{{
				Names:  denied,
				Action: seccomp.ActionErrno,
			}},
		},
	}
}

func withoutEnvironment(environment []string, name string) []string {
	prefix := name + "="
	result := make([]string, 0, len(environment))
	for _, entry := range environment {
		if len(entry) >= len(prefix) && entry[:len(prefix)] == prefix {
			continue
		}
		result = append(result, entry)
	}
	return result
}
