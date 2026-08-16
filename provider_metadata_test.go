package agentic_test

import (
	"testing"

	"github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/anthropic"
	"github.com/regularkevvv/agentic/provider/azure"
	"github.com/regularkevvv/agentic/provider/bedrock"
	"github.com/regularkevvv/agentic/provider/cohere"
	"github.com/regularkevvv/agentic/provider/deepinfra"
	"github.com/regularkevvv/agentic/provider/endpoint"
	"github.com/regularkevvv/agentic/provider/gemini"
	"github.com/regularkevvv/agentic/provider/grok"
	"github.com/regularkevvv/agentic/provider/huggingface"
	"github.com/regularkevvv/agentic/provider/ollama"
	"github.com/regularkevvv/agentic/provider/openai"
	"github.com/regularkevvv/agentic/provider/openrouter"
	"github.com/regularkevvv/agentic/provider/pinecone"
	"github.com/regularkevvv/agentic/provider/sagemaker"
	"github.com/regularkevvv/agentic/provider/together"
	"github.com/regularkevvv/agentic/provider/voyageai"
)

func TestBuiltInModelsReportSemanticProviderMetadata(t *testing.T) {
	endpoint := "https://models.example:8443/v1"
	openAI, _ := openai.New("model", openai.WithAPIKey("key"), openai.WithBaseURL(endpoint))
	responses, _ := openai.NewResponses("model", openai.WithAPIKey("key"), openai.WithBaseURL(endpoint))
	anthropicModel, _ := anthropic.New("model", anthropic.WithAPIKey("key"), anthropic.WithBaseURL(endpoint))
	azureModel, _ := azure.New("model", azure.WithAPIKey("key"), azure.WithEndpoint("https://resource.openai.azure.com"))
	bedrockModel, err := bedrock.New("model", bedrock.WithRegion("us-east-1"), bedrock.WithCredentials("key", "secret", ""))
	if err != nil {
		t.Fatalf("bedrock.New: %v", err)
	}
	geminiModel, _ := gemini.New("model", gemini.WithAPIKey("key"))
	geminiVertex, _ := gemini.New("model", gemini.WithVertexAI("project", "location"))
	grokModel, _ := grok.New("model", grok.WithAPIKey("key"), grok.WithBaseURL(endpoint))
	openRouterModel, _ := openrouter.New("model", openrouter.WithAPIKey("key"), openrouter.WithBaseURL(endpoint))
	togetherModel, _ := together.New("model", together.WithAPIKey("key"), together.WithBaseURL(endpoint))
	ollamaModel, _ := ollama.New("model", ollama.WithHost("http://localhost:11434"))

	tests := []struct {
		name     string
		provider agentic.ModelMetadataProvider
		want     string
	}{
		{"openai", openAI, "openai"},
		{"openai responses", responses, "openai"},
		{"anthropic", anthropicModel, "anthropic"},
		{"azure", azureModel, "azure.ai.openai"},
		{"bedrock", bedrockModel, "aws.bedrock"},
		{"gemini", geminiModel, "gcp.gemini"},
		{"vertex", geminiVertex, "gcp.vertex_ai"},
		{"grok", grokModel, "x_ai"},
		{"openrouter", openRouterModel, "openrouter"},
		{"together", togetherModel, "together"},
		{"ollama", ollamaModel, "ollama"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			metadata := test.provider.ModelMetadata()
			if metadata.Provider != test.want || metadata.Operation == "" {
				t.Fatalf("metadata = %#v, want provider %q and an operation", metadata, test.want)
			}
		})
	}
	if metadata := openAI.ModelMetadata(); metadata.ServerAddress != "models.example" || metadata.ServerPort != 8443 {
		t.Fatalf("OpenAI endpoint metadata = %#v", metadata)
	}
	fromClient := openai.NewResponsesFromClient("other", openAI)
	if metadata := fromClient.ModelMetadata(); metadata.ServerAddress != "models.example" {
		t.Fatalf("shared client metadata = %#v", metadata)
	}
}

func TestBuiltInEmbeddersReportSemanticProviderMetadata(t *testing.T) {
	endpointURL := "https://embeddings.example:9443/v1"
	openAI, _ := openai.NewEmbedder("model", openai.WithAPIKey("key"), openai.WithBaseURL(endpointURL))
	ollamaEmbedder, _ := ollama.NewEmbedder("model", ollama.WithHost("http://localhost:11434"))
	geminiEmbedder, _ := gemini.NewEmbedder("model", gemini.WithAPIKey("key"))
	geminiVertex, _ := gemini.NewEmbedder("model", gemini.WithVertexAI("project", "location"))
	bedrockEmbedder, err := bedrock.NewEmbedder("amazon.titan-embed-text-v2:0", bedrock.WithRegion("us-east-1"), bedrock.WithCredentials("key", "secret", ""))
	if err != nil {
		t.Fatalf("bedrock.NewEmbedder: %v", err)
	}
	cohereEmbedder, _ := cohere.New("model", cohere.WithAPIKey("key"), cohere.WithBaseURL(endpointURL))
	voyageEmbedder, _ := voyageai.New("model", voyageai.WithAPIKey("key"), voyageai.WithBaseURL(endpointURL))
	deepInfraEmbedder, _ := deepinfra.New("model", deepinfra.WithAPIToken("key"), deepinfra.WithBaseURL(endpointURL))
	endpointEmbedder, _ := endpoint.New(endpointURL, endpoint.WithoutAuthentication(), endpoint.WithModel("model"))
	huggingFaceEmbedder, _ := huggingface.NewShared("model", huggingface.WithSharedToken("key"), huggingface.WithRouterURL(endpointURL))
	pineconeEmbedder, _ := pinecone.New("model", pinecone.WithAPIKey("key"), pinecone.WithBaseURL(endpointURL))
	tests := []struct {
		provider agentic.ModelMetadataProvider
		want     string
	}{
		{openAI, "openai"},
		{ollamaEmbedder, "ollama"},
		{geminiEmbedder, "gcp.gemini"},
		{geminiVertex, "gcp.vertex_ai"},
		{bedrockEmbedder, "aws.bedrock"},
		{cohereEmbedder, "cohere"},
		{voyageEmbedder, "voyage_ai"},
		{deepInfraEmbedder, "deepinfra"},
		{endpointEmbedder, "custom"},
		{huggingFaceEmbedder, "hugging_face"},
		{pineconeEmbedder, "pinecone"},
	}
	for _, test := range tests {
		metadata := test.provider.ModelMetadata()
		if metadata.Provider != test.want || metadata.Operation != "embeddings" {
			t.Errorf("metadata = %#v, want %q embeddings", metadata, test.want)
		}
	}
	for _, provider := range []agentic.ModelMetadataProvider{
		openAI, cohereEmbedder, voyageEmbedder, deepInfraEmbedder,
		endpointEmbedder, huggingFaceEmbedder, pineconeEmbedder,
	} {
		metadata := provider.ModelMetadata()
		if metadata.ServerAddress != "embeddings.example" || metadata.ServerPort != 9443 {
			t.Errorf("embedding endpoint metadata = %#v", metadata)
		}
	}
	var _ agentic.ModelMetadataProvider = (*sagemaker.Encoder)(nil)
}
