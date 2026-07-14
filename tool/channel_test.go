package tool

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/regularkevvv/agentic/internal/core"
)

type channelInput struct {
	Value int `json:"value"`
}

func TestChannelToolExecute(t *testing.T) {
	tool, handler, err := ChannelTool("async_value", "async value", func(ctx context.Context, input channelInput) (<-chan string, error) {
		ch := make(chan string, 1)
		ch <- "value"
		close(ch)
		return ch, nil
	})
	if err != nil {
		t.Fatalf("ChannelTool: %v", err)
	}
	if tool.Function.Name != "async_value" {
		t.Fatalf("unexpected tool name %q", tool.Function.Name)
	}

	dh := handler.(*channelHandler[channelInput, string])
	if dh.Name() != "async_value" {
		t.Fatalf("unexpected handler name %q", dh.Name())
	}
	if dh.ToolConfig() != nil {
		t.Fatalf("expected nil tool config, got %#v", dh.ToolConfig())
	}

	out, err := handler.Execute(context.Background(), map[string]interface{}{"value": 7}, nil)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if out.(string) != "value" {
		t.Fatalf("unexpected output %#v", out)
	}

	t.Run("timeout-configured handler still returns immediate result", func(t *testing.T) {
		_, handler, err := ChannelTool("async_value", "async value", func(ctx context.Context, input channelInput) (<-chan string, error) {
			ch := make(chan string, 1)
			ch <- "fast"
			close(ch)
			return ch, nil
		}, WithChannelTimeout(time.Second))
		if err != nil {
			t.Fatalf("ChannelTool: %v", err)
		}

		out, execErr := handler.Execute(context.Background(), map[string]interface{}{"value": 7}, nil)
		if execErr != nil || out.(string) != "fast" {
			t.Fatalf("expected immediate result with timeout configured, got out=%#v err=%v", out, execErr)
		}
	})
}

func TestChannelToolExecuteErrors(t *testing.T) {
	t.Run("marshal input error", func(t *testing.T) {
		_, handler, err := ChannelTool("async_value", "async value", func(ctx context.Context, input channelInput) (<-chan string, error) {
			ch := make(chan string, 1)
			ch <- "ok"
			close(ch)
			return ch, nil
		})
		if err != nil {
			t.Fatalf("ChannelTool: %v", err)
		}

		_, execErr := handler.Execute(context.Background(), map[string]interface{}{"value": make(chan int)}, nil)
		if execErr == nil || execErr.Error()[:14] != "marshal input:" {
			t.Fatalf("expected marshal input error, got %v", execErr)
		}
	})

	t.Run("unmarshal input error", func(t *testing.T) {
		_, handler, err := ChannelTool("async_value", "async value", func(ctx context.Context, input channelInput) (<-chan string, error) {
			ch := make(chan string, 1)
			ch <- "ok"
			close(ch)
			return ch, nil
		})
		if err != nil {
			t.Fatalf("ChannelTool: %v", err)
		}

		_, execErr := handler.Execute(context.Background(), map[string]interface{}{"value": "bad"}, nil)
		if execErr == nil || execErr.Error()[:24] != "unmarshal to tool.channel"[:24] && execErr.Error()[:13] != "unmarshal to " {
			t.Fatalf("expected unmarshal error, got %v", execErr)
		}
	})

	t.Run("approval rejected", func(t *testing.T) {
		_, handler, err := ChannelTool("async_value", "async value", func(ctx context.Context, input channelInput) (<-chan string, error) {
			ch := make(chan string, 1)
			ch <- "value"
			close(ch)
			return ch, nil
		}, WithApproval(func(ctx context.Context, toolCall core.ToolUse) (bool, error) {
			return false, nil
		}))
		if err != nil {
			t.Fatalf("ChannelTool: %v", err)
		}

		_, execErr := handler.Execute(context.Background(), map[string]interface{}{"value": 7}, nil)
		var retryErr *core.ModelRetry
		if !errors.As(execErr, &retryErr) {
			t.Fatalf("expected modelRetry, got %T: %v", execErr, execErr)
		}
		if retryErr.Error() != `Tool "async_value" was rejected by approval` {
			t.Fatalf("unexpected retry error %q", retryErr.Error())
		}
	})

	t.Run("approval function error", func(t *testing.T) {
		_, handler, err := ChannelTool("async_value", "async value", func(ctx context.Context, input channelInput) (<-chan string, error) {
			ch := make(chan string, 1)
			ch <- "value"
			close(ch)
			return ch, nil
		}, WithApproval(func(ctx context.Context, toolCall core.ToolUse) (bool, error) {
			return false, errors.New("approval failed")
		}))
		if err != nil {
			t.Fatalf("ChannelTool: %v", err)
		}

		_, execErr := handler.Execute(context.Background(), map[string]interface{}{"value": 7}, nil)
		if execErr == nil || execErr.Error() != "approval: approval failed" {
			t.Fatalf("expected wrapped approval error, got %v", execErr)
		}
	})

	t.Run("handler returns error", func(t *testing.T) {
		_, handler, err := ChannelTool("async_value", "async value", func(ctx context.Context, input channelInput) (<-chan string, error) {
			return nil, errors.New("boom")
		})
		if err != nil {
			t.Fatalf("ChannelTool: %v", err)
		}

		_, execErr := handler.Execute(context.Background(), map[string]interface{}{"value": 7}, nil)
		if execErr == nil || execErr.Error() != "boom" {
			t.Fatalf("expected handler error, got %v", execErr)
		}
	})

	t.Run("closed channel without result", func(t *testing.T) {
		_, handler, err := ChannelTool("async_value", "async value", func(ctx context.Context, input channelInput) (<-chan string, error) {
			ch := make(chan string)
			close(ch)
			return ch, nil
		})
		if err != nil {
			t.Fatalf("ChannelTool: %v", err)
		}

		_, execErr := handler.Execute(context.Background(), map[string]interface{}{"value": 7}, nil)
		if execErr == nil || !strings.Contains(execErr.Error(), "channel closed without result") {
			t.Fatalf("expected closed channel error, got %v", execErr)
		}
	})

	t.Run("timeout", func(t *testing.T) {
		_, handler, err := ChannelTool("async_value", "async value", func(ctx context.Context, input channelInput) (<-chan string, error) {
			return make(chan string), nil
		}, WithChannelTimeout(10*time.Millisecond))
		if err != nil {
			t.Fatalf("ChannelTool: %v", err)
		}

		_, execErr := handler.Execute(context.Background(), map[string]interface{}{"value": 7}, nil)
		if execErr == nil || !strings.Contains(execErr.Error(), "timed out after") {
			t.Fatalf("expected timeout error, got %v", execErr)
		}
	})

	t.Run("context canceled", func(t *testing.T) {
		_, handler, err := ChannelTool("async_value", "async value", func(ctx context.Context, input channelInput) (<-chan string, error) {
			return make(chan string), nil
		})
		if err != nil {
			t.Fatalf("ChannelTool: %v", err)
		}

		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		_, execErr := handler.Execute(ctx, map[string]interface{}{"value": 7}, nil)
		if !errors.Is(execErr, context.Canceled) {
			t.Fatalf("expected context cancellation, got %v", execErr)
		}
	})
}

func TestMustChannelTool(t *testing.T) {
	if _, handler := MustChannelTool("must_async", "must async", func(ctx context.Context, input channelInput) (<-chan string, error) {
		ch := make(chan string, 1)
		ch <- "ok"
		close(ch)
		return ch, nil
	}); handler == nil {
		t.Fatal("expected handler")
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected MustChannelTool to panic on invalid tool")
		}
	}()
	MustChannelTool("", "desc", func(ctx context.Context, input channelInput) (<-chan string, error) {
		return nil, nil
	})
}

func TestApprovalTool(t *testing.T) {
	if _, _, err := ApprovalTool("missing_approval", "missing approval", func(context.Context, channelInput) (string, error) {
		return "", nil
	}, nil); err == nil || err.Error() != "approval function cannot be nil" {
		t.Fatalf("expected nil approval function error, got %v", err)
	}

	tool, handler, err := ApprovalTool("sync_approval", "sync approval", func(ctx context.Context, input channelInput) (string, error) {
		return "approved", nil
	}, func(ctx context.Context, toolCall core.ToolUse) (bool, error) {
		return true, nil
	})
	if err != nil {
		t.Fatalf("ApprovalTool: %v", err)
	}
	if tool.Function.Name != "sync_approval" {
		t.Fatalf("unexpected tool name %q", tool.Function.Name)
	}

	dh := handler.(*approvalHandler[channelInput, string])
	if dh.Name() != "sync_approval" {
		t.Fatalf("unexpected handler name %q", dh.Name())
	}
	if dh.ToolConfig() != nil {
		t.Fatalf("expected nil tool config, got %#v", dh.ToolConfig())
	}

	out, err := handler.Execute(context.Background(), map[string]interface{}{"value": 7}, nil)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if out.(string) != "approved" {
		t.Fatalf("unexpected output %#v", out)
	}

	_, rejectingHandler, err := ApprovalTool("sync_reject", "sync reject", func(ctx context.Context, input channelInput) (string, error) {
		return "rejected", nil
	}, func(ctx context.Context, toolCall core.ToolUse) (bool, error) {
		return false, nil
	})
	if err != nil {
		t.Fatalf("ApprovalTool: %v", err)
	}
	if _, execErr := rejectingHandler.Execute(context.Background(), map[string]interface{}{"value": 7}, nil); execErr == nil {
		t.Fatal("expected rejection error")
	}

	t.Run("approval constructor options are applied", func(t *testing.T) {
		_, handler, err := ApprovalTool("sync_approval", "sync approval", func(ctx context.Context, input channelInput) (string, error) {
			return "approved", nil
		}, func(ctx context.Context, toolCall core.ToolUse) (bool, error) {
			return true, nil
		}, WithChannelTimeout(time.Second))
		if err != nil {
			t.Fatalf("ApprovalTool: %v", err)
		}

		dh := handler.(*approvalHandler[channelInput, string])
		if dh.config == nil || dh.config.timeout != time.Second {
			t.Fatalf("expected approval config timeout to be applied, got %#v", dh.config)
		}
	})

	t.Run("approval handler unmarshal error", func(t *testing.T) {
		_, handler, err := ApprovalTool("sync_approval", "sync approval", func(ctx context.Context, input channelInput) (string, error) {
			return "approved", nil
		}, func(ctx context.Context, toolCall core.ToolUse) (bool, error) {
			return true, nil
		})
		if err != nil {
			t.Fatalf("ApprovalTool: %v", err)
		}

		_, execErr := handler.Execute(context.Background(), map[string]interface{}{"value": "bad"}, nil)
		if execErr == nil || execErr.Error()[:13] != "unmarshal to " {
			t.Fatalf("expected unmarshal error, got %v", execErr)
		}
	})

	t.Run("approval handler marshal error", func(t *testing.T) {
		_, handler, err := ApprovalTool("sync_approval", "sync approval", func(ctx context.Context, input channelInput) (string, error) {
			return "approved", nil
		}, func(ctx context.Context, toolCall core.ToolUse) (bool, error) {
			return true, nil
		})
		if err != nil {
			t.Fatalf("ApprovalTool: %v", err)
		}

		_, execErr := handler.Execute(context.Background(), map[string]interface{}{"value": func() {}}, nil)
		if execErr == nil || !strings.Contains(execErr.Error(), "marshal input") {
			t.Fatalf("expected marshal error, got %v", execErr)
		}
	})
}

func TestMustApprovalTool(t *testing.T) {
	if _, handler := MustApprovalTool("must_sync_approval", "must sync approval", func(ctx context.Context, input channelInput) (string, error) {
		return "approved", nil
	}, func(ctx context.Context, toolCall core.ToolUse) (bool, error) {
		return true, nil
	}); handler == nil {
		t.Fatal("expected handler")
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected MustApprovalTool to panic on invalid tool")
		}
	}()
	MustApprovalTool("", "desc", func(ctx context.Context, input channelInput) (string, error) {
		return "", nil
	}, func(ctx context.Context, toolCall core.ToolUse) (bool, error) {
		return true, nil
	})
}
