// Package system supplies process-local clock and cryptographic ID adapters.
package system

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"time"
)

type Clock struct{}

func NewClock() Clock { return Clock{} }

func (Clock) Now() time.Time { return time.Now() }

type IDs struct{}

func NewIDs() IDs { return IDs{} }

func (IDs) New(prefix string) (string, error) {
	if prefix == "" {
		return "", fmt.Errorf("ID prefix is required")
	}
	value := make([]byte, 16)
	if _, err := rand.Read(value); err != nil {
		return "", fmt.Errorf("generate %s ID: %w", prefix, err)
	}
	return prefix + "_" + hex.EncodeToString(value), nil
}
