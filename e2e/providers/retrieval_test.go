//go:build e2e
// +build e2e

// This file exercises the v0.3.0 retrieval providers against live APIs.
//
// The embedders and rerankers added in v0.3.0 were written against
// pydantic-ai's source and the vendored Go SDKs; their transport tests prove
// our code sends what we believe each API wants. Only these tests prove the
// endpoint paths, field names, and response shapes are actually right.
//
// Each provider skips cleanly when its key is absent:
//   - VOYAGE_API_KEY        embeddings + reranking
//   - CO_API_KEY            embeddings + reranking (COHERE_API_KEY also accepted)
//   - GEMINI_API_KEY        embeddings (GOOGLE_API_KEY also accepted)
//   - AWS credentials       embeddings (Bedrock; also needs AWS_REGION)
//   - a local Ollama server embeddings
package providers

import (
	"context"
	"os"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/e2e/internal/corpus"
	"github.com/regularkevvv/agentic/provider/bedrock"
	"github.com/regularkevvv/agentic/provider/cohere"
	"github.com/regularkevvv/agentic/provider/gemini"
	"github.com/regularkevvv/agentic/provider/ollama"
	"github.com/regularkevvv/agentic/provider/openai"
	"github.com/regularkevvv/agentic/provider/voyageai"
)

// retrievalDocs is a small corpus with one obviously correct answer, so a
// working embedder or reranker ranks it first and a broken one does not.
//
// It is not internal/corpus.Documents and must not be replaced by it: this
// query shares its wording with the answer, which is what makes a reranker's
// cross-encoding scoreable, whereas that corpus withholds the wording on
// purpose so that only a dense match succeeds.
var retrievalDocs = []string{
	"Sourdough bread is made by fermenting dough with naturally occurring lactobacilli.",
	"The monthly operating budget for the engineering team is $2,000.",
	"Go's goroutines are multiplexed onto a smaller number of OS threads.",
	"Mount Everest is the highest mountain above sea level.",
}

// budgetDocIndex is the document that answers the budget query below.
const budgetDocIndex = 1

const retrievalQuery = "what is the engineering budget?"

func skipIfNoVoyageKey(t *testing.T) {
	t.Helper()
	if os.Getenv("VOYAGE_API_KEY") == "" {
		t.Skip("VOYAGE_API_KEY not set, skipping Voyage AI e2e test")
	}
}

func skipIfNoCohereKey(t *testing.T) {
	t.Helper()
	if os.Getenv("CO_API_KEY") == "" && os.Getenv("COHERE_API_KEY") == "" {
		t.Skip("CO_API_KEY / COHERE_API_KEY not set, skipping Cohere e2e test")
	}
}

func skipIfNoGeminiKey(t *testing.T) {
	t.Helper()
	if os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY / GOOGLE_API_KEY not set, skipping Gemini e2e test")
	}
}

func skipIfNoAWSCredentials(t *testing.T) {
	t.Helper()
	hasStatic := os.Getenv("AWS_ACCESS_KEY_ID") != "" && os.Getenv("AWS_SECRET_ACCESS_KEY") != ""
	hasProfile := os.Getenv("AWS_PROFILE") != ""
	if !hasStatic && !hasProfile {
		t.Skip("AWS credentials not set, skipping Bedrock e2e test")
	}
}

func retrievalCtx(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// checkEmbedder runs the assertions every embedder must satisfy, then verifies
// the vectors are semantically useful rather than merely well-shaped.
//
// The retrieval check is the one that catches a wrong input_type or task_type
// mapping: vectors can be the right count and width and still rank the wrong
// document first if queries and documents are embedded on the wrong side of a
// retrieval-tuned model.
func checkEmbedder(t *testing.T, ctx context.Context, e agentic.Embedder) {
	t.Helper()

	docs, err := agentic.EmbedDocuments(ctx, e, retrievalDocs...)
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	if len(docs.Vectors) != len(retrievalDocs) {
		t.Fatalf("got %d vectors for %d documents", len(docs.Vectors), len(retrievalDocs))
	}

	width := len(docs.Vectors[0])
	if width == 0 {
		t.Fatal("first vector is empty")
	}
	for i, vector := range docs.Vectors {
		if vector == nil {
			t.Fatalf("vectors[%d] is nil — a response slot was never filled", i)
		}
		if len(vector) != width {
			t.Errorf("vectors[%d] has width %d, want %d — vectors are ragged", i, len(vector), width)
		}
	}
	t.Logf("embedded %d documents at %d dimensions (usage: %d tokens)",
		len(docs.Vectors), width, docs.Usage.TotalTokens)

	query, err := agentic.EmbedQuery(ctx, e, retrievalQuery)
	if err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}
	if len(query.Vectors) != 1 {
		t.Fatalf("got %d vectors for 1 query", len(query.Vectors))
	}
	if len(query.Vectors[0]) != width {
		t.Fatalf("query width %d != document width %d; they are not comparable",
			len(query.Vectors[0]), width)
	}

	best, bestScore := -1, -2.0
	for i, vector := range docs.Vectors {
		if score := corpus.Cosine(query.Vectors[0], vector); score > bestScore {
			best, bestScore = i, score
		}
	}
	if best != budgetDocIndex {
		t.Errorf("nearest document to %q was [%d] %q (score %.4f), want [%d] — check the query/document input-type mapping",
			retrievalQuery, best, retrievalDocs[best], bestScore, budgetDocIndex)
	}
	t.Logf("nearest document [%d] at cosine %.4f", best, bestScore)
}

// checkReranker runs the assertions every reranker must satisfy.
func checkReranker(t *testing.T, ctx context.Context, r agentic.Reranker) {
	t.Helper()

	resp, err := agentic.Rerank(ctx, r, retrievalQuery, retrievalDocs, 0)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(resp.Results) != len(retrievalDocs) {
		t.Fatalf("got %d results for %d documents with TopN=0", len(resp.Results), len(retrievalDocs))
	}

	for i, result := range resp.Results {
		if result.Index < 0 || result.Index >= len(retrievalDocs) {
			t.Fatalf("results[%d].Index = %d, out of range", i, result.Index)
		}
		// Document must be the caller's own text, joined by index.
		if result.Document != retrievalDocs[result.Index] {
			t.Errorf("results[%d].Document does not match the request document at index %d:\n got %q\nwant %q",
				i, result.Index, result.Document, retrievalDocs[result.Index])
		}
		if i > 0 && resp.Results[i-1].Score < result.Score {
			t.Errorf("results are not sorted by descending score: [%d]=%.4f then [%d]=%.4f",
				i-1, resp.Results[i-1].Score, i, result.Score)
		}
	}

	if resp.Results[0].Index != budgetDocIndex {
		t.Errorf("top result was [%d] %q, want [%d] %q",
			resp.Results[0].Index, resp.Results[0].Document,
			budgetDocIndex, retrievalDocs[budgetDocIndex])
	}
	t.Logf("top result [%d] score %.4f (usage: %d tokens, %d search units)",
		resp.Results[0].Index, resp.Results[0].Score, resp.Usage.TotalTokens, resp.Usage.SearchUnits)

	// TopN must actually narrow the result set.
	topped, err := agentic.Rerank(ctx, r, retrievalQuery, retrievalDocs, 2)
	if err != nil {
		t.Fatalf("Rerank with TopN=2: %v", err)
	}
	if len(topped.Results) != 2 {
		t.Errorf("TopN=2 returned %d results, want 2", len(topped.Results))
	}
}

// ============================================================================
// Embedders
// ============================================================================

func TestE2E_Retrieval_OpenAIEmbedder(t *testing.T) {
	skipIfNoOpenAIKey(t)
	e, err := openai.NewEmbedder("text-embedding-3-small")
	if err != nil {
		t.Fatalf("openai.NewEmbedder: %v", err)
	}
	checkEmbedder(t, retrievalCtx(t), e)
}

func TestE2E_Retrieval_VoyageEmbedder(t *testing.T) {
	skipIfNoVoyageKey(t)
	e, err := voyageai.New("voyage-3.5")
	if err != nil {
		t.Fatalf("voyageai.New: %v", err)
	}
	checkEmbedder(t, retrievalCtx(t), e)
}

func TestE2E_Retrieval_CohereEmbedder(t *testing.T) {
	skipIfNoCohereKey(t)
	e, err := cohere.New(cohere.DefaultEmbeddingModel)
	if err != nil {
		t.Fatalf("cohere.New: %v", err)
	}
	checkEmbedder(t, retrievalCtx(t), e)
}

func TestE2E_Retrieval_GeminiEmbedder(t *testing.T) {
	skipIfNoGeminiKey(t)
	e, err := gemini.NewEmbedder(gemini.DefaultEmbeddingModel)
	if err != nil {
		t.Fatalf("gemini.NewEmbedder: %v", err)
	}
	checkEmbedder(t, retrievalCtx(t), e)
}

// TestE2E_Retrieval_GeminiRejectsPrefixConditionedModel proves the guard that
// refuses the gemini-embedding-2 family, whose task conditioning comes from a
// text prefix rather than the taskType field this Embedder sets. Accepting one
// would silently degrade retrieval instead of failing.
func TestE2E_Retrieval_GeminiRejectsPrefixConditionedModel(t *testing.T) {
	skipIfNoGeminiKey(t)
	if _, err := gemini.NewEmbedder("gemini-embedding-2-preview"); err == nil {
		t.Fatal("NewEmbedder accepted a prefix-conditioned model; it must refuse rather than degrade silently")
	}
}

func TestE2E_Retrieval_BedrockTitanEmbedder(t *testing.T) {
	skipIfNoAWSCredentials(t)
	e, err := bedrock.NewEmbedder("amazon.titan-embed-text-v2:0")
	if err != nil {
		t.Fatalf("bedrock.NewEmbedder: %v", err)
	}
	// Titan takes one text per call, so this also exercises the internal
	// fan-out and its order preservation.
	checkEmbedder(t, retrievalCtx(t), e)
}

func TestE2E_Retrieval_OllamaEmbedder(t *testing.T) {
	skipIfNoOllama(t)
	e, err := ollama.NewEmbedder("nomic-embed-text")
	if err != nil {
		t.Fatalf("ollama.NewEmbedder: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	checkEmbedder(t, ctx, e)
}

// ============================================================================
// Rerankers
// ============================================================================

func TestE2E_Retrieval_VoyageReranker(t *testing.T) {
	skipIfNoVoyageKey(t)
	r, err := voyageai.NewReranker("rerank-2.5")
	if err != nil {
		t.Fatalf("voyageai.NewReranker: %v", err)
	}
	checkReranker(t, retrievalCtx(t), r)
}

func TestE2E_Retrieval_CohereReranker(t *testing.T) {
	skipIfNoCohereKey(t)
	r, err := cohere.NewReranker(cohere.DefaultRerankModel)
	if err != nil {
		t.Fatalf("cohere.NewReranker: %v", err)
	}
	checkReranker(t, retrievalCtx(t), r)
}

// ============================================================================
// Two-stage retrieval — the pattern the reranker exists for
// ============================================================================

// TestE2E_Retrieval_TwoStage embeds a corpus, takes the top candidates by
// vector similarity, then reorders them with a cross-encoder. It is the
// end-to-end shape a caller actually builds.
func TestE2E_Retrieval_TwoStage(t *testing.T) {
	skipIfNoVoyageKey(t)
	ctx := retrievalCtx(t)

	embedder, err := voyageai.New("voyage-3.5")
	if err != nil {
		t.Fatalf("voyageai.New: %v", err)
	}
	reranker, err := voyageai.NewReranker("rerank-2.5")
	if err != nil {
		t.Fatalf("voyageai.NewReranker: %v", err)
	}

	docs, err := agentic.EmbedDocuments(ctx, embedder, retrievalDocs...)
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	query, err := agentic.EmbedQuery(ctx, embedder, retrievalQuery)
	if err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}

	// Stage one: keep every document whose vector is at all similar, standing
	// in for the shortlist a vector store would return.
	shortlist := make([]string, 0, len(retrievalDocs))
	for i, vector := range docs.Vectors {
		if corpus.Cosine(query.Vectors[0], vector) > 0 {
			shortlist = append(shortlist, retrievalDocs[i])
		}
	}
	if len(shortlist) == 0 {
		t.Fatal("shortlist is empty; the embedding stage produced nothing to rerank")
	}

	// Stage two: reorder the shortlist with the cross-encoder.
	resp, err := agentic.Rerank(ctx, reranker, retrievalQuery, shortlist, 1)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(resp.Results) != 1 {
		t.Fatalf("got %d results, want 1", len(resp.Results))
	}
	if resp.Results[0].Document != retrievalDocs[budgetDocIndex] {
		t.Errorf("two-stage retrieval returned %q, want %q",
			resp.Results[0].Document, retrievalDocs[budgetDocIndex])
	}
	t.Logf("two-stage winner (score %.4f): %s", resp.Results[0].Score, resp.Results[0].Document)
}
