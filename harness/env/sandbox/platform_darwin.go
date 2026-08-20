//go:build darwin

package sandbox

import (
	"context"
	"errors"
	"os"

	harnessenv "github.com/regularkevvv/agentic/harness/env"
)

// Commands can read the selected workspace and the immutable system/runtime
// locations needed to start ordinary developer tools. Host home directories
// and unrelated project trees are absent. Writes are limited to the workspace
// and a fresh private temporary directory.
const seatbeltProfile = `(version 1)
(deny default)
(allow process-fork)
(allow process-exec
	(literal (param "COMMAND"))
	(literal (param "COMMAND_VARIANT"))
	(subpath (param "WORKSPACE"))
	(subpath (param "PRIVATE_TMP"))
	(subpath (param "COMMAND_HELPERS")))
(allow file-read* file-test-existence
	(literal "/")
    (subpath (param "WORKSPACE"))
    (subpath (param "PRIVATE_TMP"))
    (literal (param "COMMAND"))
	(literal (param "COMMAND_VARIANT"))
	(subpath (param "COMMAND_HELPERS"))
    (subpath "/System")
	(subpath "/usr/lib")
	(subpath "/usr/share")
	(subpath "/Library/Apple")
    (subpath "/private/etc")
    (subpath "/private/var/db")
	(literal "/private/var/select/sh")
    (subpath "/dev/fd")
    (literal "/dev/null")
    (literal "/dev/tty")
    (literal "/dev/random")
    (literal "/dev/urandom"))
(allow file-map-executable
	(literal (param "COMMAND"))
	(literal (param "COMMAND_VARIANT"))
	(subpath (param "COMMAND_HELPERS"))
	(subpath "/System/Library/Frameworks")
	(subpath "/System/Library/PrivateFrameworks")
	(subpath "/usr/lib"))
(allow file-read-data file-test-existence file-write-data
	(subpath "/dev/fd"))
(allow file-write*
    (subpath (param "WORKSPACE"))
    (subpath (param "PRIVATE_TMP"))
    (literal "/dev/null")
    (literal "/dev/tty"))
(allow sysctl-read)
`

func probeBackend() (string, error) {
	if err := validateBackendExecutable("/usr/bin/sandbox-exec"); err != nil {
		return "", err
	}
	return "seatbelt", nil
}

func validateBackendExecutable(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	if info.IsDir() || info.Mode()&0o111 == 0 {
		return errors.New(path + " is not executable")
	}
	return nil
}

func execute(ctx context.Context, value execution) (harnessenv.CommandResult, error) {
	temporary, err := privateTemp()
	if err != nil {
		return harnessenv.CommandResult{}, err
	}
	defer func() { _ = os.RemoveAll(temporary) }()
	forced := map[string]string{
		"TMPDIR": temporary, "TMP": temporary, "TEMP": temporary,
		"GOCACHE": temporary + "/go-cache", "XDG_CACHE_HOME": temporary + "/cache",
	}
	environment := commandEnvironment(value.command.Env, forced)
	target, err := resolveCommand(value.command.Name, value.cwd)
	if err != nil {
		return harnessenv.CommandResult{}, err
	}
	variant := target
	if target == "/bin/sh" {
		variant = "/bin/bash"
	}
	helpers := commandHelpers(target)
	profile := seatbeltProfile
	if value.network {
		profile += "\n(allow network*)\n"
	}
	args := []string{
		"-D", "WORKSPACE=" + value.root,
		"-D", "PRIVATE_TMP=" + temporary,
		"-D", "COMMAND=" + target,
		"-D", "COMMAND_VARIANT=" + variant,
		"-D", "COMMAND_HELPERS=" + helpers,
		"-p", profile,
		target,
	}
	args = append(args, value.command.Args...)
	return run(ctx, "/usr/bin/sandbox-exec", args, environment, value.cwd, value.command.Stdin)
}
