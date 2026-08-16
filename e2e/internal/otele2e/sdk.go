package otele2e

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"go.opentelemetry.io/otel/attribute"
	otlploggrpc "go.opentelemetry.io/otel/exporters/otlp/otlplog/otlploggrpc"
	otlpmetricgrpc "go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc"
	otlptracegrpc "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

const serviceName = "agentic-otel-e2e"

type telemetryProviders struct {
	traces  *sdktrace.TracerProvider
	metrics *sdkmetric.MeterProvider
	logs    *sdklog.LoggerProvider
}

func newTelemetryProviders(ctx context.Context, endpoint string) (*telemetryProviders, error) {
	if endpoint == "" {
		return nil, errors.New("OTLP endpoint must not be empty")
	}
	res := resource.NewSchemaless(
		attribute.String("service.name", serviceName),
		attribute.String("service.version", "e2e"),
	)

	traceExporter, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(endpoint),
		otlptracegrpc.WithInsecure(),
	)
	if err != nil {
		return nil, fmt.Errorf("create OTLP trace exporter: %w", err)
	}
	metricExporter, err := otlpmetricgrpc.New(ctx,
		otlpmetricgrpc.WithEndpoint(endpoint),
		otlpmetricgrpc.WithInsecure(),
	)
	if err != nil {
		_ = traceExporter.Shutdown(ctx)
		return nil, fmt.Errorf("create OTLP metric exporter: %w", err)
	}
	logExporter, err := otlploggrpc.New(ctx,
		otlploggrpc.WithEndpoint(endpoint),
		otlploggrpc.WithInsecure(),
	)
	if err != nil {
		_ = errors.Join(traceExporter.Shutdown(ctx), metricExporter.Shutdown(ctx))
		return nil, fmt.Errorf("create OTLP log exporter: %w", err)
	}

	return &telemetryProviders{
		traces: sdktrace.NewTracerProvider(
			sdktrace.WithResource(res),
			sdktrace.WithBatcher(traceExporter, sdktrace.WithBatchTimeout(50*time.Millisecond)),
		),
		metrics: sdkmetric.NewMeterProvider(
			sdkmetric.WithResource(res),
			sdkmetric.WithReader(sdkmetric.NewPeriodicReader(metricExporter, sdkmetric.WithInterval(50*time.Millisecond))),
		),
		logs: sdklog.NewLoggerProvider(
			sdklog.WithResource(res),
			sdklog.WithProcessor(sdklog.NewBatchProcessor(logExporter, sdklog.WithExportInterval(50*time.Millisecond))),
		),
	}, nil
}

func (p *telemetryProviders) shutdown(ctx context.Context) error {
	if p == nil {
		return nil
	}
	return errors.Join(
		p.traces.ForceFlush(ctx),
		p.metrics.ForceFlush(ctx),
		p.logs.ForceFlush(ctx),
		p.logs.Shutdown(ctx),
		p.metrics.Shutdown(ctx),
		p.traces.Shutdown(ctx),
	)
}

func waitForCollector(ctx context.Context, healthURL string) error {
	if healthURL == "" {
		return nil
	}
	client := &http.Client{Timeout: time.Second}
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	var lastErr error
	for {
		request, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
		if err != nil {
			return fmt.Errorf("build Collector health request: %w", err)
		}
		response, err := client.Do(request)
		if err == nil {
			_ = response.Body.Close()
			if response.StatusCode >= 200 && response.StatusCode < 300 {
				return nil
			}
			lastErr = fmt.Errorf("collector health status %s", response.Status)
		} else {
			lastErr = err
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("wait for Collector health at %s: %w (last error: %v)", healthURL, ctx.Err(), lastErr)
		case <-ticker.C:
		}
	}
}
