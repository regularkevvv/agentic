package permission

// ReadOnly allows filesystem reads and artifact retrieval while denying
// writes, shell, network, and unknown application tools.
func ReadOnly() *Policy {
	policy, _ := New(DecisionDeny,
		Rule{Pattern: "filesystem/read/**", Decision: DecisionAllow},
		Rule{Pattern: "filesystem/list/**", Decision: DecisionAllow},
		Rule{Pattern: "filesystem/stat/**", Decision: DecisionAllow},
		Rule{Pattern: "artifact/read/**", Decision: DecisionAllow},
	)
	return policy
}

// WorkspaceWrite allows canonical environment filesystem operations. The
// environment itself must enforce its workspace root. Shell and network remain
// ask, and unknown application tools remain denied.
func WorkspaceWrite() *Policy {
	policy, _ := New(DecisionDeny,
		Rule{Pattern: "filesystem/read/**", Decision: DecisionAllow},
		Rule{Pattern: "filesystem/list/**", Decision: DecisionAllow},
		Rule{Pattern: "filesystem/stat/**", Decision: DecisionAllow},
		Rule{Pattern: "filesystem/write/**", Decision: DecisionAllow},
		Rule{Pattern: "filesystem/mkdir/**", Decision: DecisionAllow},
		Rule{Pattern: "filesystem/remove/**", Decision: DecisionAllow},
		Rule{Pattern: "artifact/read/**", Decision: DecisionAllow},
		Rule{Pattern: "shell/**", Decision: DecisionAsk},
		Rule{Pattern: "network/**", Decision: DecisionAsk},
	)
	return policy
}
