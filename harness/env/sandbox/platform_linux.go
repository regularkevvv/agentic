//go:build linux && (amd64 || arm64)

package sandbox

import (
	"context"
	"errors"
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"

	harnessenv "github.com/regularkevvv/agentic/harness/env"
)

const (
	bootstrapArgument = "--agentic-sandbox-bootstrap"
	bootstrapEnv      = "AGENTIC_SANDBOX_BOOTSTRAP"

	landlockCreateRulesetVersion = 1
	landlockRulePathBeneath      = 1

	landlockAccessExecute    = uint64(1 << 0)
	landlockAccessWriteFile  = uint64(1 << 1)
	landlockAccessReadFile   = uint64(1 << 2)
	landlockAccessReadDir    = uint64(1 << 3)
	landlockAccessRemoveDir  = uint64(1 << 4)
	landlockAccessRemoveFile = uint64(1 << 5)
	landlockAccessMakeChar   = uint64(1 << 6)
	landlockAccessMakeDir    = uint64(1 << 7)
	landlockAccessMakeReg    = uint64(1 << 8)
	landlockAccessMakeSock   = uint64(1 << 9)
	landlockAccessMakeFIFO   = uint64(1 << 10)
	landlockAccessMakeBlock  = uint64(1 << 11)
	landlockAccessMakeSym    = uint64(1 << 12)
	landlockAccessRefer      = uint64(1 << 13)
	landlockAccessTruncate   = uint64(1 << 14)
)

type landlockRulesetAttr struct {
	HandledAccessFS uint64
}

type landlockPathBeneathAttr struct {
	AllowedAccess uint64
	ParentFD      int32
	_             uint32
}

func init() {
	if os.Getenv(bootstrapEnv) != "1" || len(os.Args) < 7 || os.Args[1] != bootstrapArgument {
		return
	}
	root, temporary, cwd, target := os.Args[2], os.Args[3], os.Args[4], os.Args[5]
	network := os.Args[6] == "network"
	if err := enterLinuxSandbox(root, temporary, target, network); err != nil {
		fmt.Fprintln(os.Stderr, "agentic sandbox:", err)
		os.Exit(126)
	}
	if err := os.Chdir(cwd); err != nil {
		fmt.Fprintln(os.Stderr, "agentic sandbox: chdir:", err)
		os.Exit(126)
	}
	environment := withoutEnvironment(os.Environ(), bootstrapEnv)
	arguments := append([]string{target}, os.Args[7:]...)
	if err := unix.Exec(target, arguments, environment); err != nil {
		fmt.Fprintln(os.Stderr, "agentic sandbox: exec:", err)
		os.Exit(126)
	}
}

func probeBackend() (string, error) {
	version, err := landlockABI()
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

func enterLinuxSandbox(root, temporary, target string, network bool) error {
	version, err := landlockABI()
	if err != nil {
		return err
	}
	handled := landlockAccessExecute | landlockAccessWriteFile | landlockAccessReadFile |
		landlockAccessReadDir | landlockAccessRemoveDir | landlockAccessRemoveFile |
		landlockAccessMakeChar | landlockAccessMakeDir | landlockAccessMakeReg |
		landlockAccessMakeSock | landlockAccessMakeFIFO | landlockAccessMakeBlock |
		landlockAccessMakeSym
	if version >= 2 {
		handled |= landlockAccessRefer
	}
	if version >= 3 {
		handled |= landlockAccessTruncate
	}
	attribute := landlockRulesetAttr{HandledAccessFS: handled}
	fd, err := landlockCreateRuleset(&attribute, unsafe.Sizeof(attribute), 0)
	if err != nil {
		return fmt.Errorf("create landlock ruleset: %w", err)
	}
	defer unix.Close(fd)
	readOnly := landlockAccessReadFile | landlockAccessReadDir
	for _, path := range []string{"/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/proc/self", "/proc/thread-self", "/opt", "/nix/store"} {
		if err := addLandlockPath(fd, path, readOnly); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	// The kernel treats an ELF interpreter as part of execution. Restrict that
	// grant to immutable system library trees rather than all system binaries.
	for _, path := range []string{"/lib", "/lib64", "/usr/lib"} {
		if err := addLandlockPath(fd, path, readOnly|landlockAccessExecute); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	helpers := commandHelpers(target)
	if err := addLandlockPath(fd, helpers, readOnly|landlockAccessExecute); err != nil {
		return err
	}
	if err := addLandlockPath(fd, target, landlockAccessReadFile|landlockAccessExecute); err != nil {
		return err
	}
	readWrite := handled
	for _, path := range []string{root, temporary} {
		if err := addLandlockPath(fd, path, readWrite); err != nil {
			return err
		}
	}
	for _, path := range []string{"/dev/null", "/dev/tty", "/dev/random", "/dev/urandom"} {
		if err := addLandlockPath(fd, path, landlockAccessReadFile|landlockAccessWriteFile); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	if err := unix.Prctl(unix.PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0); err != nil {
		return fmt.Errorf("set no-new-privileges: %w", err)
	}
	if err := landlockRestrictSelf(fd); err != nil {
		return fmt.Errorf("restrict with landlock: %w", err)
	}
	return installSeccomp(network)
}

func landlockABI() (int, error) {
	fd, err := landlockCreateRuleset(nil, 0, landlockCreateRulesetVersion)
	return fd, err
}

func landlockCreateRuleset(attribute *landlockRulesetAttr, size uintptr, flags uint32) (int, error) {
	var pointer uintptr
	if attribute != nil {
		pointer = uintptr(unsafe.Pointer(attribute))
	}
	value, _, errno := unix.Syscall(unix.SYS_LANDLOCK_CREATE_RULESET, pointer, size, uintptr(flags))
	if errno != 0 {
		return 0, errno
	}
	return int(value), nil
}

func addLandlockPath(ruleset int, path string, access uint64) error {
	fd, err := unix.Open(path, unix.O_PATH|unix.O_CLOEXEC, 0)
	if err != nil {
		return err
	}
	defer unix.Close(fd)
	attribute := landlockPathBeneathAttr{AllowedAccess: access, ParentFD: int32(fd)}
	_, _, errno := unix.Syscall6(
		unix.SYS_LANDLOCK_ADD_RULE, uintptr(ruleset), landlockRulePathBeneath,
		uintptr(unsafe.Pointer(&attribute)), 0, 0, 0,
	)
	if errno != 0 {
		return fmt.Errorf("add landlock path %s: %w", path, errno)
	}
	return nil
}

func landlockRestrictSelf(ruleset int) error {
	_, _, errno := unix.Syscall(unix.SYS_LANDLOCK_RESTRICT_SELF, uintptr(ruleset), 0, 0)
	if errno != 0 {
		return errno
	}
	return nil
}

func installSeccomp(network bool) error {
	denied := []uint32{
		unix.SYS_PTRACE, unix.SYS_MOUNT, unix.SYS_UMOUNT2, unix.SYS_PIVOT_ROOT,
		unix.SYS_CHROOT, unix.SYS_UNSHARE, unix.SYS_SETNS, unix.SYS_BPF,
		unix.SYS_PERF_EVENT_OPEN,
	}
	if !network {
		denied = append(denied, unix.SYS_SOCKET, unix.SYS_SOCKETPAIR)
	}
	filter := []unix.SockFilter{
		{Code: unix.BPF_LD | unix.BPF_W | unix.BPF_ABS, K: 4},
		{Code: unix.BPF_JMP | unix.BPF_JEQ | unix.BPF_K, Jt: 1, K: auditArchitecture},
		{Code: unix.BPF_RET | unix.BPF_K, K: unix.SECCOMP_RET_KILL_PROCESS},
		{Code: unix.BPF_LD | unix.BPF_W | unix.BPF_ABS, K: 0},
	}
	for _, number := range denied {
		filter = append(filter,
			unix.SockFilter{Code: unix.BPF_JMP | unix.BPF_JEQ | unix.BPF_K, Jf: 1, K: number},
			unix.SockFilter{Code: unix.BPF_RET | unix.BPF_K, K: unix.SECCOMP_RET_ERRNO | uint32(unix.EPERM)},
		)
	}
	filter = append(filter, unix.SockFilter{Code: unix.BPF_RET | unix.BPF_K, K: unix.SECCOMP_RET_ALLOW})
	program := unix.SockFprog{Len: uint16(len(filter)), Filter: &filter[0]}
	_, _, errno := unix.RawSyscall6(
		unix.SYS_PRCTL, unix.PR_SET_SECCOMP, unix.SECCOMP_MODE_FILTER,
		uintptr(unsafe.Pointer(&program)), 0, 0, 0,
	)
	if errno != 0 {
		return fmt.Errorf("install seccomp filter: %w", errno)
	}
	return nil
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
