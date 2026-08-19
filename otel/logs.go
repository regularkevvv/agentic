package agenticotel

import (
	"context"
	"errors"
	"time"

	"github.com/regularkevvv/agentic"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/log"
)

func (i *Instrumentation) emitException(ctx context.Context, err error) {
	if err == nil {
		return
	}
	i.emitExceptionRecord(ctx, errorType(err), err.Error())
}

func (i *Instrumentation) emitExceptionType(ctx context.Context, kind string) {
	if kind == "" {
		return
	}
	i.emitExceptionRecord(ctx, kind, "")
}

func (i *Instrumentation) emitExceptionRecord(ctx context.Context, kind, message string) {
	var record log.Record
	record.SetEventName("gen_ai.client.operation.exception")
	record.SetTimestamp(time.Now())
	record.SetSeverity(log.SeverityWarn)
	record.SetSeverityText("WARN")
	record.AddAttributes(attribute.String("exception.type", kind))
	if i.config.exceptionContent && message != "" {
		if filtered, ok := i.filtered(ContentExceptionMessage, message); ok {
			record.AddAttributes(attribute.String("exception.message", filtered))
		}
	}
	i.logger.Emit(ctx, record)
}

func (i *Instrumentation) emitInferenceDetails(ctx context.Context, operation agentic.ModelOperation, result agentic.ModelOperationResult) {
	var record log.Record
	record.SetEventName("gen_ai.client.inference.operation.details")
	record.SetTimestamp(time.Now())
	record.SetSeverity(log.SeverityInfo)
	record.SetSeverityText("INFO")
	attrs := modelAttrs(operation.Model, operation.Request.Model)
	attrs = append(attrs, requestAttrs(operation.Request)...)
	attrs = append(attrs, runAttrs(operation.Run)...)
	attrs = append(attrs, agentContextAttrs(operation.Agent)...)
	attrs = append(attrs, responseAttrs(result.Response)...)
	if result.Error != nil {
		attrs = append(attrs, attribute.String(keyErrorType, errorType(result.Error)))
	} else if result.Response != nil && result.Response.FinishReason == agentic.FinishReasonError {
		attrs = append(attrs, attribute.String(keyErrorType, "provider_error"))
	}
	if i.config.messageContent {
		system, messages := i.messageContent(operation.Request.Messages)
		if value, ok := i.contentAttribute(keySystem, system); ok && len(system) > 0 {
			attrs = append(attrs, value)
		}
		if value, ok := i.contentAttribute(keyInputMessages, messages); ok && len(messages) > 0 {
			attrs = append(attrs, value)
		}
		if result.Response != nil {
			output := i.outputMessageContent([]agentic.Message{result.Response.Message}, result.Response.FinishReason)
			if value, ok := i.contentAttribute(keyOutputMessages, output); ok && len(output) > 0 {
				attrs = append(attrs, value)
			}
		}
	}
	if i.config.toolContent && len(operation.Request.Tools) > 0 {
		if value, ok := i.contentAttribute(keyToolDefinitions, i.toolDefinitions(operation.Request.Tools)); ok {
			attrs = append(attrs, value)
		}
	}
	record.AddAttributes(attrs...)
	i.logger.Emit(ctx, record)
}

// EvaluationResult is one gen_ai.evaluation.result event. ScoreValue is a
// pointer so zero can be distinguished from an unavailable score.
type EvaluationResult struct {
	Name        string
	ScoreValue  *float64
	ScoreLabel  string
	Explanation string
	ResponseID  string
	Error       error
}

// RecordEvaluation emits a structured OTel LogRecord parented to the span in
// ctx. Evaluation happens outside Agentic's execution fold, so it is explicit
// rather than inferred from model or tool activity.
func (i *Instrumentation) RecordEvaluation(ctx context.Context, evaluation EvaluationResult) error {
	if evaluation.Name == "" {
		return errors.New("agenticotel: evaluation name must not be empty")
	}
	var record log.Record
	record.SetEventName("gen_ai.evaluation.result")
	record.SetTimestamp(time.Now())
	record.SetSeverity(log.SeverityInfo)
	record.SetSeverityText("INFO")
	attrs := []attribute.KeyValue{attribute.String("gen_ai.evaluation.name", evaluation.Name)}
	if evaluation.ScoreValue != nil {
		attrs = append(attrs, attribute.Float64("gen_ai.evaluation.score.value", *evaluation.ScoreValue))
	}
	stringAttr("gen_ai.evaluation.score.label", evaluation.ScoreLabel, &attrs)
	stringAttr(keyResponseID, evaluation.ResponseID, &attrs)
	if evaluation.Explanation != "" {
		if explanation, ok := i.filtered(ContentEvaluation, evaluation.Explanation); ok {
			attrs = append(attrs, attribute.String("gen_ai.evaluation.explanation", explanation))
		}
	}
	if evaluation.Error != nil {
		attrs = append(attrs, attribute.String(keyErrorType, errorType(evaluation.Error)))
	}
	record.AddAttributes(attrs...)
	i.logger.Emit(ctx, record)
	return nil
}
