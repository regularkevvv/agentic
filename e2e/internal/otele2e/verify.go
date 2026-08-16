package otele2e

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	agenticotel "github.com/regularkevvv/agentic/otel"

	collectorlogsv1 "go.opentelemetry.io/proto/otlp/collector/logs/v1"
	collectormetricsv1 "go.opentelemetry.io/proto/otlp/collector/metrics/v1"
	collectortracev1 "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	commonv1 "go.opentelemetry.io/proto/otlp/common/v1"
	logsv1 "go.opentelemetry.io/proto/otlp/logs/v1"
	metricsv1 "go.opentelemetry.io/proto/otlp/metrics/v1"
	tracev1 "go.opentelemetry.io/proto/otlp/trace/v1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

const (
	traceFilename  = "traces.jsonl"
	metricFilename = "metrics.jsonl"
	logFilename    = "logs.jsonl"
)

// Proof summarizes the Collector-observed signal set after every assertion.
type Proof struct {
	AdapterSpans   int
	ScenarioTraces int
	Metrics        int
	LogRecords     int
}

// WaitForProof tolerates the Collector file exporter's bounded flush interval
// while preserving the most useful assertion failure if the deadline expires.
func WaitForProof(ctx context.Context, directory string) (Proof, error) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	var lastErr error
	for {
		proof, err := Verify(directory)
		if err == nil {
			return proof, nil
		}
		lastErr = err
		select {
		case <-ctx.Done():
			return Proof{}, fmt.Errorf("wait for Collector exports in %s: %w (last assertion: %v)", directory, ctx.Err(), lastErr)
		case <-ticker.C:
		}
	}
}

// Verify parses the Collector's real OTLP JSON file-exporter output and
// asserts signal shape, topology, correlation, scenario coverage, and privacy.
func Verify(directory string) (Proof, error) {
	traceRequests, traceRaw, err := readJSONLines(filepath.Join(directory, traceFilename), func() *collectortracev1.ExportTraceServiceRequest {
		return new(collectortracev1.ExportTraceServiceRequest)
	})
	if err != nil {
		return Proof{}, err
	}
	metricRequests, metricRaw, err := readJSONLines(filepath.Join(directory, metricFilename), func() *collectormetricsv1.ExportMetricsServiceRequest {
		return new(collectormetricsv1.ExportMetricsServiceRequest)
	})
	if err != nil {
		return Proof{}, err
	}
	logRequests, logRaw, err := readJSONLines(filepath.Join(directory, logFilename), func() *collectorlogsv1.ExportLogsServiceRequest {
		return new(collectorlogsv1.ExportLogsServiceRequest)
	})
	if err != nil {
		return Proof{}, err
	}
	allRaw := append(append(append([]byte(nil), traceRaw...), metricRaw...), logRaw...)
	if bytes.Contains(allRaw, []byte(Secret)) {
		return Proof{}, errors.New("raw E2E secret leaked into Collector output")
	}
	if !bytes.Contains(allRaw, []byte(Redacted)) {
		return Proof{}, errors.New("opt-in redacted content was not exported")
	}

	spans, traceServiceSeen := collectSpans(traceRequests)
	if !traceServiceSeen {
		return Proof{}, fmt.Errorf("trace resource missing service.name=%q", serviceName)
	}
	adapterSpans := filterSpansByScope(spans, agenticotel.InstrumentationScopeName)
	if len(adapterSpans) != 16 {
		return Proof{}, fmt.Errorf("adapter span count = %d, want 16; names=%v", len(adapterSpans), sortedSpanNames(adapterSpans))
	}
	traceIDs := make(map[string]struct{})
	for _, scoped := range adapterSpans {
		traceIDs[string(scoped.span.TraceId)] = struct{}{}
	}
	if len(traceIDs) != 5 {
		return Proof{}, fmt.Errorf("scenario trace count = %d, want 5", len(traceIDs))
	}
	if err := verifyNestedTopology(spans); err != nil {
		return Proof{}, err
	}
	if err := verifySuspension(spans); err != nil {
		return Proof{}, err
	}
	if err := verifyStreaming(spans); err != nil {
		return Proof{}, err
	}
	if err := verifyEmbedding(spans); err != nil {
		return Proof{}, err
	}

	metrics, metricServiceSeen := collectMetrics(metricRequests)
	if !metricServiceSeen {
		return Proof{}, fmt.Errorf("metric resource missing service.name=%q", serviceName)
	}
	if err := verifyMetrics(metrics); err != nil {
		return Proof{}, err
	}
	logs, logServiceSeen := collectLogs(logRequests)
	if !logServiceSeen {
		return Proof{}, fmt.Errorf("log resource missing service.name=%q", serviceName)
	}
	adapterLogs := filterLogsByScope(logs, agenticotel.InstrumentationScopeName)
	if err := verifyLogs(adapterLogs, spans); err != nil {
		return Proof{}, err
	}

	return Proof{
		AdapterSpans: len(adapterSpans), ScenarioTraces: len(traceIDs), Metrics: len(metrics), LogRecords: len(adapterLogs),
	}, nil
}

func readJSONLines[T proto.Message](path string, makeMessage func() T) ([]T, []byte, error) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("open Collector export %s: %w", path, err)
	}
	var messages []T
	var raw []byte
	scanner := bufio.NewScanner(bytes.NewReader(contents))
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	line := 0
	for scanner.Scan() {
		line++
		data := bytes.TrimSpace(scanner.Bytes())
		if len(data) == 0 {
			continue
		}
		message := makeMessage()
		if err := protojson.Unmarshal(data, message); err != nil {
			return nil, nil, fmt.Errorf("decode %s line %d: %w", path, line, err)
		}
		messages = append(messages, message)
		raw = append(raw, data...)
		raw = append(raw, '\n')
	}
	if err := scanner.Err(); err != nil {
		return nil, nil, fmt.Errorf("scan Collector export %s: %w", path, err)
	}
	if len(messages) == 0 {
		return nil, nil, fmt.Errorf("collector export %s is empty", path)
	}
	return messages, raw, nil
}

type scopedSpan struct {
	scope string
	span  *tracev1.Span
}

func collectSpans(requests []*collectortracev1.ExportTraceServiceRequest) ([]scopedSpan, bool) {
	var spans []scopedSpan
	serviceSeen := false
	for _, request := range requests {
		for _, resourceSpans := range request.ResourceSpans {
			serviceSeen = serviceSeen || hasServiceName(resourceSpans.GetResource().GetAttributes())
			for _, scopeSpans := range resourceSpans.ScopeSpans {
				scope := scopeSpans.GetScope().GetName()
				for _, span := range scopeSpans.Spans {
					spans = append(spans, scopedSpan{scope: scope, span: span})
				}
			}
		}
	}
	return spans, serviceSeen
}

func filterSpansByScope(spans []scopedSpan, scope string) []scopedSpan {
	filtered := make([]scopedSpan, 0, len(spans))
	for _, span := range spans {
		if span.scope == scope {
			filtered = append(filtered, span)
		}
	}
	return filtered
}

func spansNamed(spans []scopedSpan, name string) []scopedSpan {
	var matches []scopedSpan
	for _, span := range spans {
		if span.span.Name == name {
			matches = append(matches, span)
		}
	}
	return matches
}

func oneSpan(spans []scopedSpan, name string) (scopedSpan, error) {
	matches := spansNamed(spans, name)
	if len(matches) != 1 {
		return scopedSpan{}, fmt.Errorf("span %q count = %d, want 1", name, len(matches))
	}
	return matches[0], nil
}

func sortedSpanNames(spans []scopedSpan) []string {
	names := make([]string, len(spans))
	for index, span := range spans {
		names[index] = span.span.Name
	}
	sort.Strings(names)
	return names
}

func verifyNestedTopology(spans []scopedSpan) error {
	outer, err := oneSpan(spans, "invoke_agent orchestrator")
	if err != nil {
		return err
	}
	delegate, err := oneSpan(spans, "execute_tool delegate")
	if err != nil {
		return err
	}
	child, err := oneSpan(spans, "invoke_agent researcher")
	if err != nil {
		return err
	}
	childModel, err := oneSpan(spans, "chat e2e-child-model")
	if err != nil {
		return err
	}
	outerModels := spansNamed(spans, "chat e2e-outer-model")
	if len(outerModels) != 2 {
		return fmt.Errorf("outer model span count = %d, want 2", len(outerModels))
	}
	if !sameID(delegate.span.ParentSpanId, outer.span.SpanId) ||
		!sameID(child.span.ParentSpanId, delegate.span.SpanId) ||
		!sameID(childModel.span.ParentSpanId, child.span.SpanId) {
		return errors.New("nested agent/tool span parentage is incorrect")
	}
	for _, model := range outerModels {
		if !sameID(model.span.ParentSpanId, outer.span.SpanId) || !sameID(model.span.TraceId, outer.span.TraceId) {
			return errors.New("outer model span is not a child of the outer invocation")
		}
	}
	if !sameID(outer.span.TraceId, delegate.span.TraceId) || !sameID(outer.span.TraceId, child.span.TraceId) {
		return errors.New("nested topology crossed trace IDs")
	}
	outerAttrs := attributes(outer.span.Attributes)
	delegateAttrs := attributes(delegate.span.Attributes)
	if outerAttrs["gen_ai.conversation.id"].GetStringValue() != "conversation-nested" ||
		outerAttrs["agentic.run.id"].GetStringValue() != "run-nested" {
		return errors.New("nested invocation correlation attributes are missing")
	}
	for _, key := range []string{"gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.tool.definitions"} {
		if _, ok := outerAttrs[key]; !ok {
			return fmt.Errorf("opt-in invocation content attribute %q is missing", key)
		}
	}
	for _, key := range []string{"gen_ai.tool.call.arguments", "gen_ai.tool.call.result"} {
		if _, ok := delegateAttrs[key]; !ok {
			return fmt.Errorf("opt-in tool content attribute %q is missing", key)
		}
	}
	if outer.span.Kind != tracev1.Span_SPAN_KIND_INTERNAL || delegate.span.Kind != tracev1.Span_SPAN_KIND_INTERNAL || childModel.span.Kind != tracev1.Span_SPAN_KIND_CLIENT {
		return errors.New("nested scenario span kinds are incorrect")
	}
	return nil
}

func verifySuspension(spans []scopedSpan) error {
	invocations := spansNamed(spans, "invoke_agent approval-agent")
	if len(invocations) != 2 {
		return fmt.Errorf("approval invocation count = %d, want 2", len(invocations))
	}
	var start, resume *tracev1.Span
	for _, invocation := range invocations {
		attrs := attributes(invocation.span.Attributes)
		if attrs["gen_ai.conversation.id"].GetStringValue() != "conversation-approval" || attrs["agentic.run.id"].GetStringValue() != "run-approval" {
			return errors.New("approval invocation lost durable correlation IDs")
		}
		switch attrs["agentic.execution.mode"].GetStringValue() {
		case "start":
			start = invocation.span
			if attrs["agentic.execution.outcome"].GetStringValue() != "suspended" {
				return errors.New("start invocation did not record suspended outcome")
			}
		case "resume":
			resume = invocation.span
			if attrs["agentic.execution.outcome"].GetStringValue() != "completed" {
				return errors.New("resume invocation did not record completed outcome")
			}
		}
	}
	if start == nil || resume == nil {
		return errors.New("approval start/resume invocation modes are incomplete")
	}
	tool, err := oneSpan(spans, "execute_tool approved_action")
	if err != nil {
		return err
	}
	if !sameID(tool.span.ParentSpanId, resume.SpanId) || sameID(tool.span.ParentSpanId, start.SpanId) {
		return errors.New("approved tool did not execute under the resume invocation")
	}
	return nil
}

func verifyStreaming(spans []scopedSpan) error {
	model, err := oneSpan(spans, "chat e2e-stream-model")
	if err != nil {
		return err
	}
	agent, err := oneSpan(spans, "invoke_agent streaming-agent")
	if err != nil {
		return err
	}
	modelAttrs := attributes(model.span.Attributes)
	if !modelAttrs["gen_ai.request.stream"].GetBoolValue() {
		return errors.New("streaming inference span omitted gen_ai.request.stream")
	}
	if modelAttrs["gen_ai.response.time_to_first_chunk"].GetDoubleValue() <= 0 {
		return errors.New("streaming inference span omitted positive TTFC")
	}
	if _, exists := attributes(agent.span.Attributes)["gen_ai.request.stream"]; exists {
		return errors.New("invoke_agent span incorrectly included client-only stream attribute")
	}
	return nil
}

func verifyEmbedding(spans []scopedSpan) error {
	span, err := oneSpan(spans, "embeddings e2e-embedder")
	if err != nil {
		return err
	}
	attrs := attributes(span.span.Attributes)
	if attrs["gen_ai.operation.name"].GetStringValue() != "embeddings" ||
		attrs["gen_ai.provider.name"].GetStringValue() != "e2e" ||
		attrs["gen_ai.response.model"].GetStringValue() != "e2e-embedder-v1" ||
		attrs["gen_ai.embeddings.dimension.count"].GetIntValue() != 4 ||
		attrs["gen_ai.usage.input_tokens"].GetIntValue() != 9 {
		return errors.New("embedding span semantic attributes are incomplete")
	}
	return nil
}

type scopedMetric struct {
	scope  string
	metric *metricsv1.Metric
}

func collectMetrics(requests []*collectormetricsv1.ExportMetricsServiceRequest) (map[string][]scopedMetric, bool) {
	metrics := make(map[string][]scopedMetric)
	serviceSeen := false
	for _, request := range requests {
		for _, resourceMetrics := range request.ResourceMetrics {
			serviceSeen = serviceSeen || hasServiceName(resourceMetrics.GetResource().GetAttributes())
			for _, scopeMetrics := range resourceMetrics.ScopeMetrics {
				scope := scopeMetrics.GetScope().GetName()
				for _, metric := range scopeMetrics.Metrics {
					if scope == agenticotel.InstrumentationScopeName {
						metrics[metric.Name] = append(metrics[metric.Name], scopedMetric{scope: scope, metric: metric})
					}
				}
			}
		}
	}
	return metrics, serviceSeen
}

func verifyMetrics(metrics map[string][]scopedMetric) error {
	expected := []string{
		"gen_ai.client.token.usage",
		"gen_ai.client.operation.duration",
		"gen_ai.client.operation.time_to_first_chunk",
		"gen_ai.client.operation.time_per_output_chunk",
		"gen_ai.invoke_agent.duration",
		"gen_ai.invoke_agent.inference_calls",
		"gen_ai.invoke_agent.tool_calls",
		"gen_ai.execute_tool.duration",
	}
	if len(metrics) != len(expected) {
		return fmt.Errorf("unique adapter metric count = %d, want %d; names=%v", len(metrics), len(expected), sortedMetricNames(metrics))
	}
	for _, name := range expected {
		observations := metrics[name]
		if len(observations) == 0 {
			return fmt.Errorf("metric %q is missing", name)
		}
		points := 0
		for _, observation := range observations {
			if histogram := observation.metric.GetHistogram(); histogram != nil {
				points += len(histogram.DataPoints)
			}
		}
		if points == 0 {
			return fmt.Errorf("metric %q has no histogram data points", name)
		}
	}
	return nil
}

func sortedMetricNames(metrics map[string][]scopedMetric) []string {
	names := make([]string, 0, len(metrics))
	for name := range metrics {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

type scopedLog struct {
	scope string
	log   *logsv1.LogRecord
}

func collectLogs(requests []*collectorlogsv1.ExportLogsServiceRequest) ([]scopedLog, bool) {
	var logs []scopedLog
	serviceSeen := false
	for _, request := range requests {
		for _, resourceLogs := range request.ResourceLogs {
			serviceSeen = serviceSeen || hasServiceName(resourceLogs.GetResource().GetAttributes())
			for _, scopeLogs := range resourceLogs.ScopeLogs {
				scope := scopeLogs.GetScope().GetName()
				for _, record := range scopeLogs.LogRecords {
					logs = append(logs, scopedLog{scope: scope, log: record})
				}
			}
		}
	}
	return logs, serviceSeen
}

func filterLogsByScope(logs []scopedLog, scope string) []scopedLog {
	var filtered []scopedLog
	for _, record := range logs {
		if record.scope == scope {
			filtered = append(filtered, record)
		}
	}
	return filtered
}

func verifyLogs(logs []scopedLog, spans []scopedSpan) error {
	if len(logs) != 5 {
		return fmt.Errorf("adapter log count = %d, want 5", len(logs))
	}
	byEvent := make(map[string][]*logsv1.LogRecord)
	for _, scoped := range logs {
		if len(normalizedID(scoped.log.TraceId)) != 32 || len(normalizedID(scoped.log.SpanId)) != 16 {
			return fmt.Errorf("log %q is not trace/span correlated", scoped.log.EventName)
		}
		byEvent[scoped.log.EventName] = append(byEvent[scoped.log.EventName], scoped.log)
	}
	if len(byEvent["gen_ai.client.inference.operation.details"]) != 3 ||
		len(byEvent["gen_ai.client.operation.exception"]) != 1 ||
		len(byEvent["gen_ai.evaluation.result"]) != 1 {
		return fmt.Errorf("log event counts = inference:%d exception:%d evaluation:%d, want 3/1/1",
			len(byEvent["gen_ai.client.inference.operation.details"]),
			len(byEvent["gen_ai.client.operation.exception"]),
			len(byEvent["gen_ai.evaluation.result"]),
		)
	}
	errorSpan, err := oneSpan(spans, "chat e2e-error-model")
	if err != nil {
		return err
	}
	if errorSpan.span.GetStatus().GetCode() != tracev1.Status_STATUS_CODE_ERROR {
		return errors.New("provider error span status is not ERROR")
	}
	exception := byEvent["gen_ai.client.operation.exception"][0]
	if !sameID(exception.TraceId, errorSpan.span.TraceId) || !sameID(exception.SpanId, errorSpan.span.SpanId) {
		return errors.New("exception log is not correlated to the provider error span")
	}
	exceptionAttrs := attributes(exception.Attributes)
	if exceptionAttrs["exception.type"].GetStringValue() == "" {
		return errors.New("exception log omitted exception.type")
	}
	if _, exists := exceptionAttrs["exception.message"]; exists {
		return errors.New("default-private exception log exported exception.message")
	}
	evaluationAttrs := attributes(byEvent["gen_ai.evaluation.result"][0].Attributes)
	if evaluationAttrs["gen_ai.evaluation.name"].GetStringValue() != "correctness" ||
		evaluationAttrs["gen_ai.evaluation.score.label"].GetStringValue() != "pass" ||
		!strings.Contains(evaluationAttrs["gen_ai.evaluation.explanation"].GetStringValue(), Redacted) {
		return errors.New("evaluation log attributes are incomplete or unredacted")
	}
	return nil
}

func attributes(values []*commonv1.KeyValue) map[string]*commonv1.AnyValue {
	result := make(map[string]*commonv1.AnyValue, len(values))
	for _, value := range values {
		result[value.Key] = value.Value
	}
	return result
}

func hasServiceName(values []*commonv1.KeyValue) bool {
	return attributes(values)["service.name"].GetStringValue() == serviceName
}

func sameID(left, right []byte) bool {
	leftID, rightID := normalizedID(left), normalizedID(right)
	return leftID != "" && leftID == rightID
}

// Collector's OTLP JSON file exporter uses the OTLP/JSON hexadecimal encoding
// for trace_id and span_id. protojson treats bytes as base64, so it decodes the
// hexadecimal text as if it were base64. Re-encoding recovers that text. The
// normal protobuf JSON base64 representation is accepted too, which keeps this
// verifier robust if the file exporter changes representation.
func normalizedID(value []byte) string {
	switch len(value) {
	case 8, 16:
		return hex.EncodeToString(value)
	}
	candidate := base64.StdEncoding.EncodeToString(value)
	if len(candidate) != 16 && len(candidate) != 32 {
		return ""
	}
	if _, err := hex.DecodeString(candidate); err != nil {
		return ""
	}
	return strings.ToLower(candidate)
}
