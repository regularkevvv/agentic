package agenticotel

import (
	"context"
	"errors"
	"time"

	"github.com/regularkevvv/agentic"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
)

type embedderConfig struct {
	metadata *agentic.ModelMetadata
}

// EmbedderOption configures one instrumented embedding provider.
type EmbedderOption func(*embedderConfig) error

// WithEmbedderMetadata overrides metadata reported by the embedder. Provider
// packages may implement agentic.ModelMetadataProvider; use this option for a
// custom or compatible embedding endpoint.
func WithEmbedderMetadata(metadata agentic.ModelMetadata) EmbedderOption {
	return func(c *embedderConfig) error {
		copy := metadata
		c.metadata = &copy
		return nil
	}
}

// WrapEmbedder decorates an Agentic Embedder with the official embeddings
// client span and shared GenAI client duration/token metrics.
func (i *Instrumentation) WrapEmbedder(embedder agentic.Embedder, options ...EmbedderOption) (agentic.Embedder, error) {
	if embedder == nil {
		return nil, errors.New("agenticotel: embedder must not be nil")
	}
	configuration := embedderConfig{}
	for _, option := range options {
		if option == nil {
			return nil, errors.New("agenticotel: embedder option must not be nil")
		}
		if err := option(&configuration); err != nil {
			return nil, err
		}
	}
	metadata := agentic.ModelMetadata{Provider: "custom", Operation: "embeddings"}
	if provider, ok := embedder.(agentic.ModelMetadataProvider); ok {
		reported := provider.ModelMetadata()
		if reported.Provider != "" {
			metadata.Provider = reported.Provider
		}
		metadata.ServerAddress = reported.ServerAddress
		metadata.ServerPort = reported.ServerPort
		metadata.InProcess = reported.InProcess
	}
	if configuration.metadata != nil {
		metadata = *configuration.metadata
	}
	metadata.Operation = "embeddings"
	if metadata.Provider == "" {
		metadata.Provider = "custom"
	}
	return &instrumentedEmbedder{instrumentation: i, embedder: embedder, metadata: metadata}, nil
}

type instrumentedEmbedder struct {
	instrumentation *Instrumentation
	embedder        agentic.Embedder
	metadata        agentic.ModelMetadata
}

func (e *instrumentedEmbedder) Name() string { return e.embedder.Name() }

func (e *instrumentedEmbedder) Embed(ctx context.Context, request *agentic.EmbeddingRequest) (*agentic.EmbeddingResponse, error) {
	start := time.Now()
	attrs := modelAttrs(e.metadata, e.embedder.Name())
	kind := trace.SpanKindClient
	if e.metadata.InProcess {
		kind = trace.SpanKindInternal
	}
	ctx, span := e.instrumentation.tracer.Start(ctx, "embeddings "+e.embedder.Name(),
		trace.WithTimestamp(start),
		trace.WithSpanKind(kind),
		trace.WithAttributes(attrs...),
	)
	response, err := e.embedder.Embed(ctx, request)
	end := time.Now()
	defer span.End(trace.WithTimestamp(end))
	errorKind := ""
	if err != nil {
		errorKind = markSpanError(span, err, e.instrumentation.config.exceptionContent)
		e.instrumentation.emitException(ctx, err)
	}
	metricAttrs := append([]attribute.KeyValue(nil), attrs...)
	if response != nil {
		stringAttr(keyResponseModel, response.Model, &metricAttrs)
		if response.Model != "" {
			span.SetAttributes(attribute.String(keyResponseModel, response.Model))
		}
		if response.Usage.PromptTokens > 0 {
			span.SetAttributes(attribute.Int(keyInputTokens, response.Usage.PromptTokens))
			tokenAttrs := append(append([]attribute.KeyValue(nil), metricAttrs...), attribute.String("gen_ai.token.type", "input"))
			e.instrumentation.clientTokenUsage.Record(ctx, int64(response.Usage.PromptTokens), metric.WithAttributes(tokenAttrs...))
		}
		if len(response.Vectors) > 0 && len(response.Vectors[0]) > 0 {
			span.SetAttributes(attribute.Int("gen_ai.embeddings.dimension.count", len(response.Vectors[0])))
		}
	}
	if errorKind != "" {
		metricAttrs = append(metricAttrs, attribute.String(keyErrorType, errorKind))
	}
	e.instrumentation.clientDuration.Record(ctx, end.Sub(start).Seconds(), metric.WithAttributes(metricAttrs...))
	return response, err
}

var _ agentic.Embedder = (*instrumentedEmbedder)(nil)
