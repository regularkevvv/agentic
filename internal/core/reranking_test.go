package core

import "testing"

func TestRerankRequestValidate(t *testing.T) {
	tests := []struct {
		name    string
		req     RerankRequest
		wantErr string
	}{
		{
			name: "valid single document",
			req:  RerankRequest{Query: "q", Documents: []string{"a"}},
		},
		{
			name: "valid batch with top n",
			req:  RerankRequest{Query: "q", Documents: []string{"a", "b"}, TopN: 1},
		},
		{
			name: "valid top n zero means all",
			req:  RerankRequest{Query: "q", Documents: []string{"a", "b"}},
		},
		{
			name: "valid top n larger than documents",
			req:  RerankRequest{Query: "q", Documents: []string{"a"}, TopN: 10},
		},
		{
			name:    "empty query",
			req:     RerankRequest{Documents: []string{"a"}},
			wantErr: "query cannot be empty",
		},
		{
			name:    "empty documents",
			req:     RerankRequest{Query: "q"},
			wantErr: "documents cannot be empty",
		},
		{
			name:    "empty string in documents",
			req:     RerankRequest{Query: "q", Documents: []string{"a", ""}},
			wantErr: "documents cannot be empty strings",
		},
		{
			name:    "negative top n",
			req:     RerankRequest{Query: "q", Documents: []string{"a"}, TopN: -1},
			wantErr: "top n must be non-negative",
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
