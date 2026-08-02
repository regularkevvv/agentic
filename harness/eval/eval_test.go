package eval

import (
	"bytes"
	"context"
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestRunnerStableOrderingUnderConcurrencyAndAggregation(t *testing.T) {
	subject := SubjectFunc[int, string](func(ctx context.Context, current Case[int]) Outcome[string] {
		select {
		case <-ctx.Done():
			return Outcome[string]{Error: ctx.Err()}
		case <-time.After(time.Duration(4-current.Input) * time.Millisecond):
			return Outcome[string]{Output: current.ID + "-output"}
		}
	})
	runner, err := NewRunner(subject, Config{Concurrency: 3, SampleTimeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	evaluators := []NamedEvaluator[int, string]{
		{ID: "contains", Evaluator: EvaluatorFunc[int, string](func(_ context.Context, current Case[int], outcome Outcome[string]) (Score, error) {
			return BooleanScore(strings.Contains(outcome.Output, current.ID), "contains case ID"), nil
		})},
		{ID: "length", Evaluator: EvaluatorFunc[int, string](func(_ context.Context, _ Case[int], outcome Outcome[string]) (Score, error) {
			value := float64(len(outcome.Output))
			return Score{Available: true, Passed: true, Value: &value}, nil
		})},
	}
	report, err := runner.Run(context.Background(), []Case[int]{
		{ID: "first", Input: 1, Samples: 2},
		{ID: "second", Input: 3, Samples: 1},
	}, evaluators)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]string, len(report.Results))
	for index, result := range report.Results {
		got[index] = result.CaseID + "/" + string(rune('0'+result.Sample)) + "/" + result.EvaluatorID
	}
	want := []string{"first/0/contains", "first/0/length", "first/1/contains", "first/1/length", "second/0/contains", "second/0/length"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("result order = %v", got)
	}
	if len(report.Aggregates) != 2 || report.Aggregates[0].Passed != 3 || report.Aggregates[0].Mean == nil ||
		report.Aggregates[1].Available != 3 || report.Aggregates[1].Mean == nil {
		t.Fatalf("aggregates = %#v", report.Aggregates)
	}
	if report.Results[0].Outcome.Duration <= 0 {
		t.Fatal("sample duration was not captured")
	}
}

func TestRunnerCancellationTimeoutAndEvaluatorErrorsAreHonest(t *testing.T) {
	blocking := SubjectFunc[string, string](func(ctx context.Context, _ Case[string]) Outcome[string] {
		<-ctx.Done()
		return Outcome[string]{Error: ctx.Err()}
	})
	runner, _ := NewRunner(blocking, Config{Concurrency: 1, SampleTimeout: 10 * time.Millisecond})
	var evaluatorCalls atomic.Int32
	report, err := runner.Run(context.Background(), []Case[string]{{ID: "timeout", Samples: 1}}, []NamedEvaluator[string, string]{
		{ID: "never", Evaluator: EvaluatorFunc[string, string](func(context.Context, Case[string], Outcome[string]) (Score, error) {
			evaluatorCalls.Add(1)
			return BooleanScore(true, ""), nil
		})},
	})
	if err != nil || evaluatorCalls.Load() != 0 || report.Results[0].EvaluatorError == "" || report.Aggregates[0].Errors != 1 {
		t.Fatalf("timeout report=%#v err=%v evaluatorCalls=%d", report, err, evaluatorCalls.Load())
	}

	ready := SubjectFunc[string, string](func(context.Context, Case[string]) Outcome[string] { return Outcome[string]{Output: "value"} })
	runner, _ = NewRunner(ready, Config{Concurrency: 1})
	report, err = runner.Run(context.Background(), []Case[string]{{ID: "errors", Samples: 1}}, []NamedEvaluator[string, string]{
		{ID: "error", Evaluator: EvaluatorFunc[string, string](func(context.Context, Case[string], Outcome[string]) (Score, error) {
			return Score{}, errors.New("scorer failed")
		})},
		{ID: "missing", Evaluator: EvaluatorFunc[string, string](func(context.Context, Case[string], Outcome[string]) (Score, error) {
			return Score{}, nil
		})},
	})
	if err != nil || report.Results[0].EvaluatorError != "scorer failed" || report.Results[1].EvaluatorError == "" {
		t.Fatalf("evaluator errors = %#v, %v", report.Results, err)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	var canceledSubjectCalls atomic.Int32
	canceledRunner, _ := NewRunner(SubjectFunc[string, string](func(context.Context, Case[string]) Outcome[string] {
		canceledSubjectCalls.Add(1)
		return Outcome[string]{Output: "should not run"}
	}), Config{Concurrency: 1})
	report, err = canceledRunner.Run(canceled, []Case[string]{{ID: "canceled", Samples: 2}}, []NamedEvaluator[string, string]{{ID: "score", Evaluator: ExpectError[string, string](true)}})
	if err != nil || len(report.Results) != 2 || report.Results[0].Outcome.ErrorMessage != context.Canceled.Error() || canceledSubjectCalls.Load() != 0 {
		t.Fatalf("canceled report = %#v, %v; subject calls = %d", report, err, canceledSubjectCalls.Load())
	}
}

func TestBuiltinsReporterAndValidation(t *testing.T) {
	current := Case[int]{ID: "case", Expected: "hello", Samples: 1}
	for name, evaluator := range map[string]Evaluator[int, string]{
		"exact":     ExactText[int](),
		"substring": ContainsText[int](),
	} {
		t.Run(name, func(t *testing.T) {
			score, err := evaluator.Evaluate(context.Background(), current, Outcome[string]{Output: "hello"})
			if err != nil || !score.Passed || score.Value == nil || *score.Value != 1 {
				t.Fatalf("score = %#v, %v", score, err)
			}
		})
	}
	score, err := ExpectError[int, string](true).Evaluate(context.Background(), current, Outcome[string]{Error: errors.New("expected")})
	if err != nil || !score.Passed {
		t.Fatalf("error score = %#v, %v", score, err)
	}
	if _, err := ExactText[int]().Evaluate(context.Background(), Case[int]{Expected: 1}, Outcome[string]{}); err == nil {
		t.Fatal("non-string expectation succeeded")
	}
	report := Report[int, string]{Version: 1, Cases: []Case[int]{current}}
	var first, second bytes.Buffer
	if err := WriteJSON(&first, report); err != nil {
		t.Fatal(err)
	}
	_ = WriteJSON(&second, report)
	if first.String() != second.String() || !strings.Contains(first.String(), `"version":1`) {
		t.Fatalf("unstable JSON: %q %q", first.String(), second.String())
	}
	if WriteJSON[int, string](nil, report) == nil || WriteJSON(&bytes.Buffer{}, Report[int, string]{Version: 2}) == nil {
		t.Fatal("invalid reporter input succeeded")
	}
	if _, err := NewRunner[string, string](nil, Config{}); err == nil {
		t.Fatal("nil subject succeeded")
	}
	if _, err := NewRunner(SubjectFunc[string, string](func(context.Context, Case[string]) Outcome[string] { return Outcome[string]{} }), Config{Concurrency: -1}); err == nil {
		t.Fatal("invalid concurrency succeeded")
	}
	runner, _ := NewRunner(SubjectFunc[string, string](func(context.Context, Case[string]) Outcome[string] { return Outcome[string]{} }), Config{})
	evaluator := NamedEvaluator[string, string]{ID: "score", Evaluator: ExpectError[string, string](false)}
	for _, cases := range [][]Case[string]{nil, {{ID: "", Samples: 1}}, {{ID: "same", Samples: 1}, {ID: "same", Samples: 1}}, {{ID: "zero", Samples: 0}}} {
		if _, err := runner.Run(context.Background(), cases, []NamedEvaluator[string, string]{evaluator}); err == nil {
			t.Fatalf("invalid cases %#v succeeded", cases)
		}
	}
	if _, err := runner.Run(context.Background(), []Case[string]{{ID: "case", Samples: 1}}, []NamedEvaluator[string, string]{{ID: "same", Evaluator: evaluator.Evaluator}, {ID: "same", Evaluator: evaluator.Evaluator}}); err == nil {
		t.Fatal("duplicate evaluator succeeded")
	}
}

func TestBuiltinErrorAndAggregationFrontiers(t *testing.T) {
	current := Case[int]{ID: "case", Expected: "expected", Samples: 1}
	for name, evaluator := range map[string]Evaluator[int, string]{
		"exact error":    ExactText[int](),
		"contains error": ContainsText[int](),
	} {
		t.Run(name, func(t *testing.T) {
			score, err := evaluator.Evaluate(context.Background(), current, Outcome[string]{Error: errors.New("subject failed")})
			if err != nil || score.Passed || score.Reason != "subject failed" {
				t.Fatalf("score = %#v, %v", score, err)
			}
		})
	}
	if _, err := ContainsText[int]().Evaluate(context.Background(), Case[int]{Expected: 42}, Outcome[string]{}); err == nil {
		t.Fatal("substring evaluator accepted non-string expectation")
	}
	values := aggregate([]Result[int, string]{{CaseID: "case", EvaluatorID: "missing"}}, []NamedEvaluator[int, string]{{ID: "missing", Evaluator: ExpectError[int, string](false)}})
	if len(values) != 1 || values[0].Count != 1 || values[0].Available != 0 {
		t.Fatalf("unavailable aggregate = %#v", values)
	}

	timeoutSubject := SubjectFunc[int, string](func(ctx context.Context, _ Case[int]) Outcome[string] {
		<-ctx.Done()
		return Outcome[string]{Output: "late"}
	})
	runner, err := NewRunner(timeoutSubject, Config{Concurrency: 1, SampleTimeout: time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	report, err := runner.Run(context.Background(), []Case[int]{{ID: "timeout", Samples: 1}}, []NamedEvaluator[int, string]{{ID: "score", Evaluator: ExactText[int]()}})
	if err != nil || !errors.Is(report.Results[0].Outcome.Error, context.DeadlineExceeded) {
		t.Fatalf("implicit timeout outcome = %#v, %v", report, err)
	}
}
