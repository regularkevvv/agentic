package eval

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

type Config struct {
	Concurrency   int
	SampleTimeout time.Duration
}

type Runner[I, O any] struct {
	subject Subject[I, O]
	config  Config
}

func NewRunner[I, O any](subject Subject[I, O], config Config) (*Runner[I, O], error) {
	if subject == nil {
		return nil, errors.New("eval subject is required")
	}
	if config.Concurrency == 0 {
		config.Concurrency = 1
	}
	if config.Concurrency < 1 || config.SampleTimeout < 0 {
		return nil, errors.New("eval concurrency must be positive and timeout non-negative")
	}
	return &Runner[I, O]{subject: subject, config: config}, nil
}

type sampleTask[I any] struct {
	index  int
	caseID int
	sample int
	value  Case[I]
}

func (r *Runner[I, O]) Run(
	ctx context.Context,
	cases []Case[I],
	evaluators []NamedEvaluator[I, O],
) (Report[I, O], error) {
	if err := validateInputs(cases, evaluators); err != nil {
		return Report[I, O]{}, err
	}
	caseCopies := append([]Case[I](nil), cases...)
	tasks := make([]sampleTask[I], 0)
	for caseIndex, current := range caseCopies {
		for sample := 0; sample < current.Samples; sample++ {
			tasks = append(tasks, sampleTask[I]{index: len(tasks), caseID: caseIndex, sample: sample, value: current})
		}
	}
	perTask := make([][]Result[I, O], len(tasks))
	semaphore := make(chan struct{}, r.config.Concurrency)
	var wait sync.WaitGroup
	for _, task := range tasks {
		task := task
		wait.Add(1)
		go func() {
			defer wait.Done()
			select {
			case semaphore <- struct{}{}:
				defer func() { <-semaphore }()
			case <-ctx.Done():
				perTask[task.index] = unavailableResults(task, evaluators, ctx.Err())
				return
			}
			perTask[task.index] = r.runSample(ctx, task, evaluators)
		}()
	}
	wait.Wait()
	report := Report[I, O]{Version: 1, Cases: caseCopies}
	for _, values := range perTask {
		report.Results = append(report.Results, values...)
	}
	report.Aggregates = aggregate(report.Results, evaluators)
	return report, nil
}

func (r *Runner[I, O]) runSample(
	ctx context.Context,
	task sampleTask[I],
	evaluators []NamedEvaluator[I, O],
) []Result[I, O] {
	sampleCtx := ctx
	cancel := func() {}
	if r.config.SampleTimeout > 0 {
		sampleCtx, cancel = context.WithTimeout(ctx, r.config.SampleTimeout)
	}
	defer cancel()
	started := time.Now()
	outcome := r.subject.Run(sampleCtx, task.value)
	if outcome.Duration == 0 {
		outcome.Duration = time.Since(started)
	}
	if outcome.Error == nil && sampleCtx.Err() != nil {
		outcome.Error = sampleCtx.Err()
	}
	if outcome.Error != nil && outcome.ErrorMessage == "" {
		outcome.ErrorMessage = outcome.Error.Error()
	}
	results := make([]Result[I, O], len(evaluators))
	for index, named := range evaluators {
		result := Result[I, O]{
			CaseID: task.value.ID, Sample: task.sample, EvaluatorID: named.ID, Outcome: outcome,
		}
		if sampleCtx.Err() != nil {
			result.EvaluatorError = sampleCtx.Err().Error()
			results[index] = result
			continue
		}
		score, err := named.Evaluator.Evaluate(sampleCtx, task.value, outcome)
		if err != nil {
			result.EvaluatorError = err.Error()
		} else if !score.Available {
			result.EvaluatorError = "evaluator returned an unavailable score without an error"
		} else {
			result.Score = cloneScore(score)
		}
		results[index] = result
	}
	return results
}

func validateInputs[I, O any](cases []Case[I], evaluators []NamedEvaluator[I, O]) error {
	if len(cases) == 0 || len(evaluators) == 0 {
		return errors.New("eval cases and evaluators are required")
	}
	caseIDs := make(map[string]bool, len(cases))
	for _, current := range cases {
		if current.ID == "" || strings.TrimSpace(current.ID) != current.ID || caseIDs[current.ID] || current.Samples <= 0 {
			return fmt.Errorf("invalid or duplicate eval case %q", current.ID)
		}
		caseIDs[current.ID] = true
	}
	evaluatorIDs := make(map[string]bool, len(evaluators))
	for _, current := range evaluators {
		if current.ID == "" || strings.TrimSpace(current.ID) != current.ID || evaluatorIDs[current.ID] || current.Evaluator == nil {
			return fmt.Errorf("invalid or duplicate eval evaluator %q", current.ID)
		}
		evaluatorIDs[current.ID] = true
	}
	return nil
}

func unavailableResults[I, O any](
	task sampleTask[I],
	evaluators []NamedEvaluator[I, O],
	cause error,
) []Result[I, O] {
	var zero O
	outcome := Outcome[O]{Output: zero, Error: cause, ErrorMessage: cause.Error()}
	results := make([]Result[I, O], len(evaluators))
	for index, evaluator := range evaluators {
		results[index] = Result[I, O]{
			CaseID: task.value.ID, Sample: task.sample, EvaluatorID: evaluator.ID,
			Outcome: outcome, EvaluatorError: cause.Error(),
		}
	}
	return results
}

func aggregate[I, O any](results []Result[I, O], evaluators []NamedEvaluator[I, O]) []Aggregate {
	values := make([]Aggregate, len(evaluators))
	index := make(map[string]int, len(evaluators))
	sums := make([]float64, len(evaluators))
	numeric := make([]int, len(evaluators))
	for position, evaluator := range evaluators {
		values[position].EvaluatorID = evaluator.ID
		index[evaluator.ID] = position
	}
	for _, result := range results {
		position := index[result.EvaluatorID]
		values[position].Count++
		if result.EvaluatorError != "" {
			values[position].Errors++
			continue
		}
		if !result.Score.Available {
			continue
		}
		values[position].Available++
		if result.Score.Passed {
			values[position].Passed++
		}
		if result.Score.Value != nil {
			sums[position] += *result.Score.Value
			numeric[position]++
		}
	}
	for position := range values {
		if numeric[position] > 0 {
			mean := sums[position] / float64(numeric[position])
			values[position].Mean = &mean
		}
	}
	return values
}

func cloneScore(score Score) Score {
	if score.Value != nil {
		value := *score.Value
		score.Value = &value
	}
	return score
}
