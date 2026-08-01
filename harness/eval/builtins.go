package eval

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

func ExactText[I any]() Evaluator[I, string] {
	return EvaluatorFunc[I, string](func(_ context.Context, current Case[I], outcome Outcome[string]) (Score, error) {
		expected, ok := current.Expected.(string)
		if !ok {
			return Score{}, errors.New("exact-text evaluator requires a string expected value")
		}
		if outcome.Error != nil {
			return BooleanScore(false, outcome.Error.Error()), nil
		}
		return BooleanScore(outcome.Output == expected, fmt.Sprintf("expected %q", expected)), nil
	})
}

func ContainsText[I any]() Evaluator[I, string] {
	return EvaluatorFunc[I, string](func(_ context.Context, current Case[I], outcome Outcome[string]) (Score, error) {
		expected, ok := current.Expected.(string)
		if !ok {
			return Score{}, errors.New("substring evaluator requires a string expected value")
		}
		if outcome.Error != nil {
			return BooleanScore(false, outcome.Error.Error()), nil
		}
		return BooleanScore(strings.Contains(outcome.Output, expected), fmt.Sprintf("expected substring %q", expected)), nil
	})
}

func ExpectError[I, O any](expected bool) Evaluator[I, O] {
	return EvaluatorFunc[I, O](func(_ context.Context, _ Case[I], outcome Outcome[O]) (Score, error) {
		actual := outcome.Error != nil
		return BooleanScore(actual == expected, fmt.Sprintf("expected error=%t, got error=%t", expected, actual)), nil
	})
}
