//go:build linux && arm64

package sandbox

import "golang.org/x/sys/unix"

const auditArchitecture = unix.AUDIT_ARCH_AARCH64
