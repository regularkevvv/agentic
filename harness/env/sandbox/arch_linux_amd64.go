//go:build linux && amd64

package sandbox

import "golang.org/x/sys/unix"

const auditArchitecture = unix.AUDIT_ARCH_X86_64
