package agenticotel

import (
	"errors"

	"go.opentelemetry.io/otel/log"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
)

// ContentKind identifies a potentially sensitive value passed to a filter.
type ContentKind string

const (
	ContentMessageText      ContentKind = "message_text"
	ContentReasoning        ContentKind = "reasoning"
	ContentURI              ContentKind = "uri"
	ContentToolArguments    ContentKind = "tool_arguments"
	ContentToolResult       ContentKind = "tool_result"
	ContentToolDescription  ContentKind = "tool_description"
	ContentToolParameters   ContentKind = "tool_parameters"
	ContentFileID           ContentKind = "file_id"
	ContentExceptionMessage ContentKind = "exception_message"
	ContentEvaluation       ContentKind = "evaluation_explanation"
)

// ContentFilter may redact or reject one sensitive string. Returning false
// omits the containing field. Filters run only for explicitly enabled content.
type ContentFilter func(ContentKind, string) (string, bool)

type config struct {
	tracerProvider   trace.TracerProvider
	meterProvider    metric.MeterProvider
	loggerProvider   log.LoggerProvider
	messageContent   bool
	toolContent      bool
	exceptionContent bool
	inferenceDetails bool
	filter           ContentFilter
	maxContentBytes  int
}

// Option configures Instrumentation.
type Option func(*config) error

func WithTracerProvider(provider trace.TracerProvider) Option {
	return func(c *config) error {
		if provider == nil {
			return errors.New("agenticotel: tracer provider must not be nil")
		}
		c.tracerProvider = provider
		return nil
	}
}

func WithMeterProvider(provider metric.MeterProvider) Option {
	return func(c *config) error {
		if provider == nil {
			return errors.New("agenticotel: meter provider must not be nil")
		}
		c.meterProvider = provider
		return nil
	}
}

func WithLoggerProvider(provider log.LoggerProvider) Option {
	return func(c *config) error {
		if provider == nil {
			return errors.New("agenticotel: logger provider must not be nil")
		}
		c.loggerProvider = provider
		return nil
	}
}

// WithMessageContent enables opt-in message, system-instruction, and model
// output content on spans and inference-detail log records.
func WithMessageContent() Option {
	return func(c *config) error { c.messageContent = true; return nil }
}

// WithToolContent enables opt-in tool definitions, arguments, and results.
func WithToolContent() Option {
	return func(c *config) error { c.toolContent = true; return nil }
}

// WithExceptionContent enables exception messages on exception log records
// and conventional exception events on spans. It can contain provider data.
func WithExceptionContent() Option {
	return func(c *config) error { c.exceptionContent = true; return nil }
}

// WithInferenceDetails emits the opt-in
// gen_ai.client.inference.operation.details log record. Content fields still
// require WithMessageContent or WithToolContent.
func WithInferenceDetails() Option {
	return func(c *config) error { c.inferenceDetails = true; return nil }
}

// WithContentFilter installs an application privacy policy for enabled
// content fields.
func WithContentFilter(filter ContentFilter) Option {
	return func(c *config) error {
		if filter == nil {
			return errors.New("agenticotel: content filter must not be nil")
		}
		c.filter = filter
		return nil
	}
}

// WithMaxContentBytes omits an entire structured content attribute when its
// valid JSON representation exceeds max bytes. Zero means no adapter limit.
func WithMaxContentBytes(max int) Option {
	return func(c *config) error {
		if max < 0 {
			return errors.New("agenticotel: max content bytes must be non-negative")
		}
		c.maxContentBytes = max
		return nil
	}
}
