package core

import "testing"

func TestEmbeddingRequestValidate(t *testing.T) {
	tests := []struct {
		name    string
		req     EmbeddingRequest
		wantErr string
	}{
		{
			name: "valid single input",
			req:  EmbeddingRequest{Input: []string{"hello"}},
		},
		{
			name: "valid batch with query type and dimensions",
			req: EmbeddingRequest{
				Input:      []string{"a", "b"},
				InputType:  EmbeddingInputQuery,
				Dimensions: 512,
			},
		},
		{
			name: "valid document type",
			req:  EmbeddingRequest{Input: []string{"a"}, InputType: EmbeddingInputDocument},
		},
		{
			name:    "empty input",
			req:     EmbeddingRequest{},
			wantErr: "input cannot be empty",
		},
		{
			name:    "empty string in input",
			req:     EmbeddingRequest{Input: []string{"a", ""}},
			wantErr: "input texts cannot be empty strings",
		},
		{
			name:    "unknown input type",
			req:     EmbeddingRequest{Input: []string{"a"}, InputType: "passage"},
			wantErr: "input type must be query, document, or empty",
		},
		{
			name:    "negative dimensions",
			req:     EmbeddingRequest{Input: []string{"a"}, Dimensions: -1},
			wantErr: "dimensions must be non-negative",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.req.Validate()
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("Validate() = %v, want nil", err)
				}
				return
			}
			if err == nil || err.Error() != tt.wantErr {
				t.Fatalf("Validate() = %v, want %q", err, tt.wantErr)
			}
		})
	}
}
