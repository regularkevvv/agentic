package agentic

import (
	"context"
	"testing"
)

func TestWithTemperature(t *testing.T) {
	cfg := defaultAgentConfig()
	WithTemperature(0.7)(&cfg)
	if cfg.temperature == nil || *cfg.temperature != 0.7 {
		t.Errorf("expected temperature 0.7, got %v", cfg.temperature)
	}
}

func TestWithMaxTokens(t *testing.T) {
	cfg := defaultAgentConfig()
	WithMaxTokens(500)(&cfg)
	if cfg.maxTokens == nil || *cfg.maxTokens != 500 {
		t.Errorf("expected max tokens 500, got %v", cfg.maxTokens)
	}
}

func TestWithTopP(t *testing.T) {
	cfg := defaultAgentConfig()
	WithTopP(0.9)(&cfg)
	if cfg.topP == nil || *cfg.topP != 0.9 {
		t.Errorf("expected topP 0.9, got %v", cfg.topP)
	}
}

func TestWithToolChoice(t *testing.T) {
	cfg := defaultAgentConfig()
	WithToolChoice(ToolChoiceRequired)(&cfg)
	if cfg.toolChoice == nil || *cfg.toolChoice != ToolChoiceRequired {
		t.Errorf("expected tool choice required, got %v", cfg.toolChoice)
	}
}

func TestWithMessages(t *testing.T) {
	opts := applyRunOptions([]RunOption{
		WithMessages(NewTextMessage(RoleUser, "hello")),
	})
	if len(opts.messages) != 1 {
		t.Errorf("expected 1 message, got %d", len(opts.messages))
	}
}

func TestWithRunTemperature(t *testing.T) {
	opts := applyRunOptions([]RunOption{
		WithRunTemperature(0.5),
	})
	if opts.temperature == nil || *opts.temperature != 0.5 {
		t.Errorf("expected temperature 0.5, got %v", opts.temperature)
	}
}

func TestWithRunMaxTokens(t *testing.T) {
	opts := applyRunOptions([]RunOption{
		WithRunMaxTokens(200),
	})
	if opts.maxTokens == nil || *opts.maxTokens != 200 {
		t.Errorf("expected max tokens 200, got %v", opts.maxTokens)
	}
}

func TestWithRunMaxIterations(t *testing.T) {
	opts := applyRunOptions([]RunOption{
		WithRunMaxIterations(5),
	})
	if opts.maxIterations == nil || *opts.maxIterations != 5 {
		t.Errorf("expected max iterations 5, got %v", opts.maxIterations)
	}
}

func TestPromptCacheOptionsCopyValues(t *testing.T) {
	config := PromptCacheConfig{Key: "agent", Retention: PromptCacheLong}
	agentConfig := defaultAgentConfig()
	WithPromptCache(config)(&agentConfig)
	config.Key = "mutated"
	if agentConfig.promptCache == nil || agentConfig.promptCache.Key != "agent" {
		t.Fatalf("agent prompt cache = %#v", agentConfig.promptCache)
	}
	runConfig := PromptCacheConfig{Key: "run", Retention: PromptCacheShort}
	options := applyRunOptions([]RunOption{WithRunPromptCache(runConfig)})
	runConfig.Key = "mutated"
	if options.promptCache == nil || options.promptCache.Key != "run" {
		t.Fatalf("run prompt cache = %#v", options.promptCache)
	}
}

func TestDefaultAgentConfig(t *testing.T) {
	cfg := defaultAgentConfig()
	if cfg.maxIterations != 10 {
		t.Errorf("expected default maxIterations 10, got %d", cfg.maxIterations)
	}
	if cfg.retryConfig.MaxRetries != 1 {
		t.Errorf("expected default maxRetries 1, got %d", cfg.retryConfig.MaxRetries)
	}
}

func TestHistoryProcessorOptions(t *testing.T) {
	proc := HistoryProcessorFunc(func(ctx context.Context, messages []Message) ([]Message, error) {
		return append(messages, NewTextMessage(RoleAssistant, "processed")), nil
	})

	cfg := defaultAgentConfig()
	WithHistoryProcessor(proc)(&cfg)
	if cfg.historyProcessor == nil {
		t.Fatal("expected agent history processor to be set")
	}

	opts := applyRunOptions([]RunOption{WithRunHistoryProcessor(proc)})
	if opts.historyProcessor == nil {
		t.Fatal("expected run history processor to be set")
	}

	got, err := opts.historyProcessor.Process(context.Background(), []Message{NewTextMessage(RoleUser, "hello")})
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if len(got) != 2 || got[1].GetTextContent() != "processed" {
		t.Fatalf("unexpected processed messages %#v", got)
	}
}
