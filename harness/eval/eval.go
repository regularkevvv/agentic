// Package eval provides deterministic, generic evaluation values and runners.
// It owns no model, storage, file, or network implementation.
package eval

import (
	"context"
	"time"
)

type Case[I any] struct {
	ID       string `json:"id"`
	Input    I      `json:"input"`
	Expected any    `json:"expected,omitempty"`
	Samples  int    `json:"samples"`
}

type Outcome[O any] struct {
	Output       O             `json:"output"`
	Error        error         `json:"-"`
	ErrorMessage string        `json:"error,omitempty"`
	Duration     time.Duration `json:"duration"`
}

type Score struct {
	Available bool     `json:"available"`
	Passed    bool     `json:"passed"`
	Value     *float64 `json:"value,omitempty"`
	Reason    string   `json:"reason,omitempty"`
}

func BooleanScore(passed bool, reason string) Score {
	value := 0.0
	if passed {
		value = 1
	}
	return Score{Available: true, Passed: passed, Value: &value, Reason: reason}
}

type Subject[I, O any] interface {
	Run(context.Context, Case[I]) Outcome[O]
}

type SubjectFunc[I, O any] func(context.Context, Case[I]) Outcome[O]

func (f SubjectFunc[I, O]) Run(ctx context.Context, current Case[I]) Outcome[O] {
	return f(ctx, current)
}

type Evaluator[I, O any] interface {
	Evaluate(context.Context, Case[I], Outcome[O]) (Score, error)
}

type EvaluatorFunc[I, O any] func(context.Context, Case[I], Outcome[O]) (Score, error)

func (f EvaluatorFunc[I, O]) Evaluate(ctx context.Context, current Case[I], outcome Outcome[O]) (Score, error) {
	return f(ctx, current, outcome)
}

type NamedEvaluator[I, O any] struct {
	ID        string
	Evaluator Evaluator[I, O]
}

type Result[I, O any] struct {
	CaseID         string     `json:"case_id"`
	Sample         int        `json:"sample"`
	EvaluatorID    string     `json:"evaluator_id"`
	Outcome        Outcome[O] `json:"outcome"`
	Score          Score      `json:"score"`
	EvaluatorError string     `json:"evaluator_error,omitempty"`
}

type Aggregate struct {
	EvaluatorID string   `json:"evaluator_id"`
	Count       int      `json:"count"`
	Available   int      `json:"available"`
	Passed      int      `json:"passed"`
	Errors      int      `json:"errors"`
	Mean        *float64 `json:"mean,omitempty"`
}

type Report[I, O any] struct {
	Version    int            `json:"version"`
	Cases      []Case[I]      `json:"cases"`
	Results    []Result[I, O] `json:"results"`
	Aggregates []Aggregate    `json:"aggregates"`
}
