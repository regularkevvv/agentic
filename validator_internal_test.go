package agentic

import (
	"context"
	"errors"
	"strings"
	"testing"
)

type validatorCoverageValue struct {
	Value string `json:"value"`
}

func TestValidateStructAdditionalTagsAndTypedValidator(t *testing.T) {
	tests := []struct {
		name   string
		value  any
		substr string
	}{
		{
			name: "len",
			value: struct {
				Code string `json:"code" validate:"len=3"`
			}{Code: "abcd"},
			substr: "code must have exactly 3 elements",
		},
		{
			name: "email",
			value: struct {
				Email string `json:"email" validate:"email"`
			}{Email: "not-an-email"},
			substr: "email must be a valid email address",
		},
		{
			name: "url",
			value: struct {
				URL string `json:"url" validate:"url"`
			}{URL: "not-a-url"},
			substr: "url must be a valid URL",
		},
		{
			name: "contains",
			value: struct {
				Body string `json:"body" validate:"contains=ok"`
			}{Body: "missing"},
			substr: "body must contain 'ok'",
		},
		{
			name: "gt",
			value: struct {
				Count int `json:"count" validate:"gt=5"`
			}{Count: 5},
			substr: "count must be greater than 5",
		},
		{
			name: "gte",
			value: struct {
				Count int `json:"count" validate:"gte=5"`
			}{Count: 4},
			substr: "count must be greater than or equal to 5",
		},
		{
			name: "lt",
			value: struct {
				Count int `json:"count" validate:"lt=5"`
			}{Count: 5},
			substr: "count must be less than 5",
		},
		{
			name: "lte",
			value: struct {
				Count int `json:"count" validate:"lte=5"`
			}{Count: 6},
			substr: "count must be less than or equal to 5",
		},
		{
			name: "default branch",
			value: struct {
				Name string `json:"name" validate:"startswith=ab"`
			}{Name: "zz"},
			substr: "name failed 'startswith' validation",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateStruct(tt.value)
			if err == nil || !strings.Contains(err.Error(), tt.substr) {
				t.Fatalf("expected error containing %q, got %v", tt.substr, err)
			}
		})
	}

	validator := TypedOutputValidatorFunc[validatorCoverageValue](func(ctx context.Context, output validatorCoverageValue) error {
		if output.Value != "ok" {
			return errors.New("unexpected output")
		}
		return nil
	})
	if err := validator.ValidateTyped(context.Background(), validatorCoverageValue{Value: "ok"}); err != nil {
		t.Fatalf("expected typed validator to pass, got %v", err)
	}
}

func TestValidateStructPointerAndFieldNameFallbacks(t *testing.T) {
	tests := []struct {
		name   string
		value  any
		substr string
	}{
		{
			name: "pointer input uses json field name",
			value: &struct {
				DisplayName string `json:"display_name" validate:"required"`
			}{},
			substr: "display_name is required",
		},
		{
			name: "dash json tag falls back to Go name",
			value: struct {
				Hidden string `json:"-" validate:"required"`
			}{},
			substr: "Hidden is required",
		},
		{
			name: "missing json tag falls back to Go name",
			value: struct {
				PlainField string `validate:"required"`
			}{},
			substr: "PlainField is required",
		},
		{
			name: "string min message",
			value: struct {
				Body string `json:"body" validate:"min=4"`
			}{Body: "hey"},
			substr: "body must be at least 4 characters long",
		},
		{
			name: "string max message",
			value: struct {
				Body string `json:"body" validate:"max=2"`
			}{Body: "long"},
			substr: "body must be at most 2 characters long",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateStruct(tt.value)
			if err == nil || !strings.Contains(err.Error(), tt.substr) {
				t.Fatalf("expected error containing %q, got %v", tt.substr, err)
			}
		})
	}
}
