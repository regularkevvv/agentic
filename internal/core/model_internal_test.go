package core

import "testing"

func TestModelMetadataForEndpoint(t *testing.T) {
	tests := []struct {
		name     string
		endpoint string
		address  string
		port     int
	}{
		{name: "host and port", endpoint: "https://models.example:8443/v1", address: "models.example", port: 8443},
		{name: "https default port", endpoint: "https://models.example/v1", address: "models.example", port: 443},
		{name: "http default port", endpoint: "http://models.example/v1", address: "models.example", port: 80},
		{name: "empty", endpoint: ""},
		{name: "invalid", endpoint: "://bad"},
		{name: "invalid port", endpoint: "https://models.example:bad/v1"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			metadata := ModelMetadataForEndpoint("provider", "chat", test.endpoint)
			if metadata.Provider != "provider" || metadata.Operation != "chat" ||
				metadata.ServerAddress != test.address || metadata.ServerPort != test.port {
				t.Fatalf("metadata = %#v", metadata)
			}
		})
	}
}
