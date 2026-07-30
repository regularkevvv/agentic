package subagent

import (
	"context"
	"fmt"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/harness"
)

// Children returns stable addresses for child sessions that are currently
// executing a delegation call. Pass an empty parent ID to inspect all live
// routes. Completed children remain durable but are not retained in this
// process-local routing table.
func (c *Capability) Children(parentSessionID string) []Address {
	if c == nil || c.router == nil {
		return nil
	}
	return c.router.addresses(parentSessionID)
}

func (c *Capability) Steer(ctx context.Context, address Address, message agentic.Message) (harness.QueueReceipt, error) {
	child, err := c.addressedChild(address)
	if err != nil {
		return harness.QueueReceipt{}, err
	}
	return child.Steer(ctx, message)
}

func (c *Capability) FollowUp(ctx context.Context, address Address, message agentic.Message) (harness.QueueReceipt, error) {
	child, err := c.addressedChild(address)
	if err != nil {
		return harness.QueueReceipt{}, err
	}
	return child.FollowUp(ctx, message)
}

func (c *Capability) NextTurn(ctx context.Context, address Address, message agentic.Message) (harness.QueueReceipt, error) {
	child, err := c.addressedChild(address)
	if err != nil {
		return harness.QueueReceipt{}, err
	}
	return child.NextTurn(ctx, message)
}

func (c *Capability) Interrupt(ctx context.Context, address Address) error {
	child, err := c.addressedChild(address)
	if err != nil {
		return err
	}
	return child.Interrupt(ctx)
}

func (c *Capability) Snapshot(ctx context.Context, address Address) (harness.Snapshot, error) {
	child, err := c.addressedChild(address)
	if err != nil {
		return harness.Snapshot{}, err
	}
	return child.Snapshot(ctx)
}

func (c *Capability) addressedChild(address Address) (childControl, error) {
	if c == nil || c.router == nil {
		return nil, fmt.Errorf("%w: topology is not initialized", ErrChildNotFound)
	}
	return c.router.child(address)
}
