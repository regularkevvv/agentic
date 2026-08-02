//go:build e2e
// +build e2e

// This file exercises the multi-representation encoders against live APIs.
//
// The transport tests prove that each provider sends what we believe its API
// wants and decodes what we believe it returns. Only these tests prove that
// the resulting vectors retrieve: that a dense paraphrase match, a sparse rare
// term match, and a late-interaction match each put the right document first.
// Asserting HTTP 200 would prove none of that.
//
// Each provider skips cleanly when its gate is absent:
//
//   - DeepInfra native BGE-M3: DEEPINFRA_TOKEN
//   - Hugging Face shared router (dense only): HF_TOKEN, HF_SHARED_MODEL
//   - Hugging Face dedicated endpoint: HF_TOKEN, HF_ENDPOINT_URL, HF_DENSE_SPACE_ID
//   - SageMaker endpoint: AWS credentials, AWS_REGION, SAGEMAKER_ENDPOINT,
//     SAGEMAKER_DENSE_SPACE_ID
//   - Pinecone Inference: PINECONE_API_KEY, PINECONE_SPARSE_MODEL
//
// The dedicated and SageMaker gates include the expected space ID on purpose.
// A live test that accepted whatever identity the endpoint reported would pass
// against a redeployment onto different weights, which is the failure the
// space descriptor exists to catch.
//
// The corpus is deliberately tiny. These are metered APIs, and a handful of
// short documents is enough to separate a working encoder from a broken one.
package e2e

import (
	"context"
	"errors"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	agentic "github.com/regularkevvv/agentic"
	"github.com/regularkevvv/agentic/provider/deepinfra"
	"github.com/regularkevvv/agentic/provider/huggingface"
	"github.com/regularkevvv/agentic/provider/pinecone"
	"github.com/regularkevvv/agentic/provider/sagemaker"
)

// representationDocs has one document per retrieval mode being tested.
//
// budgetDoc is the paraphrase target: it answers the query without sharing its
// wording, so only a dense match finds it. quenselDoc carries a coined term
// that appears in no other document and in no vocabulary, so only a lexical
// match finds it.
var representationDocs = []string{
	"Sourdough bread is made by fermenting dough with naturally occurring lactobacilli.",
	"Engineering spends two thousand dollars a month running its infrastructure.",
	"The quensel actuator must be recalibrated before every third launch.",
	"Mount Everest is the highest mountain above sea level.",
}

const (
	budgetDoc  = 1
	quenselDoc = 2

	paraphraseQuery = "how much does the engineering team cost each month?"
	rareTermQuery   = "quensel actuator recalibration"

	// expansionQuery is short and ordinary on purpose: a learned sparse model
	// that expands will weigh terms it never contained.
	expansionQuery = "automobile"
)

func skipIfNoDeepInfraToken(t *testing.T) {
	t.Helper()
	if os.Getenv("DEEPINFRA_TOKEN") == "" {
		t.Skip("DEEPINFRA_TOKEN not set, skipping DeepInfra e2e test")
	}
}

func representationCtx(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// dotProduct scores a sparse query against a sparse document by summing the
// weights of the coordinates they share. This is what a sparse index computes;
// doing it here proves the returned coordinates line up without needing one.
func dotProduct(query, document *agentic.SparseVector) float64 {
	var score float64
	i, j := 0, 0
	for i < len(query.Indices) && j < len(document.Indices) {
		switch {
		case query.Indices[i] < document.Indices[j]:
			i++
		case query.Indices[i] > document.Indices[j]:
			j++
		default:
			score += float64(query.Values[i]) * float64(document.Values[j])
			i++
			j++
		}
	}
	return score
}

// maxSim is ColBERT's late-interaction score: for each query token take its
// best-matching document token, then sum. It is the reason multi-vector output
// must never be averaged into one vector.
func maxSim(query, document [][]float32) float64 {
	var total float64
	for _, q := range query {
		best := math.Inf(-1)
		for _, d := range document {
			if score := cosine(q, d); score > best {
				best = score
			}
		}
		if !math.IsInf(best, -1) {
			total += best
		}
	}
	return total
}

// bestBy returns the index of the document scoring highest under score.
func bestBy(t *testing.T, count int, score func(i int) float64) int {
	t.Helper()
	best, bestScore := -1, math.Inf(-1)
	for i := range count {
		s := score(i)
		t.Logf("document [%d] scores %.4f", i, s)
		if s > bestScore {
			best, bestScore = i, s
		}
	}
	return best
}

func TestE2E_Representations_DeepInfraDenseAndSparse(t *testing.T) {
	skipIfNoDeepInfraToken(t)
	ctx := representationCtx(t)

	encoder, err := deepinfra.New(deepinfra.BGEM3Model)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// One request for both kinds. That it is one request is the point: BGE-M3
	// computes them in a single forward pass, and issuing two calls would pay
	// twice for the same work.
	docs, err := agentic.EncodeDocuments(ctx, encoder, representationDocs,
		agentic.RepresentationDense, agentic.RepresentationSparse)
	if err != nil {
		t.Fatalf("EncodeDocuments: %v", err)
	}
	if len(docs.Data) != len(representationDocs) {
		t.Fatalf("got %d representations for %d documents", len(docs.Data), len(representationDocs))
	}
	if docs.Usage.RequestCount != 1 {
		t.Errorf("request count = %d, want one call for both kinds", docs.Usage.RequestCount)
	}
	if docs.Usage.InputTokens < 0 || docs.Usage.InputBytes < 0 {
		t.Errorf("usage has a negative measurement: %+v", docs.Usage)
	}

	denseSpace, ok := docs.Space(agentic.RepresentationDense)
	if !ok {
		t.Fatal("response describes no dense space")
	}
	sparseSpace, ok := docs.Space(agentic.RepresentationSparse)
	if !ok {
		t.Fatal("response describes no sparse space")
	}
	t.Logf("dense space %s", denseSpace)
	t.Logf("sparse space %s", sparseSpace)

	if denseSpace.Dimensions != 1024 {
		t.Errorf("dense width = %d; BGE-M3 is documented at 1024", denseSpace.Dimensions)
	}
	if sparseSpace.Metric != agentic.SimilarityDotProduct {
		t.Errorf("sparse metric = %q, want dot_product", sparseSpace.Metric)
	}
	if denseSpace.ID == sparseSpace.ID {
		t.Error("dense and sparse share a space ID; they cannot share an index")
	}

	// Every document must carry both kinds, or a partial batch has been
	// accepted somewhere.
	for i, item := range docs.Data {
		if len(item.Dense) != denseSpace.Dimensions {
			t.Fatalf("document %d dense width = %d", i, len(item.Dense))
		}
		if item.Sparse.Len() == 0 {
			t.Fatalf("document %d has no sparse coordinates", i)
		}
	}

	queries, err := agentic.EncodeQueries(ctx, encoder, []string{paraphraseQuery, rareTermQuery},
		agentic.RepresentationDense, agentic.RepresentationSparse)
	if err != nil {
		t.Fatalf("EncodeQueries: %v", err)
	}

	// BGE-M3 is symmetric: the query and document roles encode into the same
	// space, so the two sides are directly comparable.
	querySpace, _ := queries.Space(agentic.RepresentationDense)
	if querySpace.ID != denseSpace.ID {
		t.Fatalf("query space %s differs from document space %s", querySpace.ID, denseSpace.ID)
	}

	t.Run("dense finds the paraphrase", func(t *testing.T) {
		best := bestBy(t, len(docs.Data), func(i int) float64 {
			return cosine(queries.Data[0].Dense, docs.Data[i].Dense)
		})
		if best != budgetDoc {
			t.Errorf("nearest document = %d, want %d (the paraphrase)", best, budgetDoc)
		}
	})

	t.Run("sparse finds the rare term", func(t *testing.T) {
		best := bestBy(t, len(docs.Data), func(i int) float64 {
			return dotProduct(queries.Data[1].Sparse, docs.Data[i].Sparse)
		})
		if best != quenselDoc {
			t.Errorf("highest sparse score = %d, want %d (the rare term)", best, quenselDoc)
		}
	})
}

func TestE2E_Representations_DeepInfraMultiVector(t *testing.T) {
	skipIfNoDeepInfraToken(t)
	ctx := representationCtx(t)

	// Multi-vector responses are large, so this runs against two documents
	// rather than the whole corpus.
	encoder, err := deepinfra.New(deepinfra.BGEM3Model)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	corpus := []string{representationDocs[0], representationDocs[budgetDoc]}
	docs, err := agentic.EncodeDocuments(ctx, encoder, corpus, agentic.RepresentationMultiVector)
	if err != nil {
		t.Fatalf("EncodeDocuments: %v", err)
	}
	queries, err := agentic.EncodeQueries(ctx, encoder, []string{paraphraseQuery}, agentic.RepresentationMultiVector)
	if err != nil {
		t.Fatalf("EncodeQueries: %v", err)
	}

	space, _ := docs.Space(agentic.RepresentationMultiVector)
	t.Logf("multi-vector space %s", space)

	// Token count varies with the document; token width must not.
	for i, item := range docs.Data {
		if len(item.MultiVector) == 0 {
			t.Fatalf("document %d has no token vectors", i)
		}
		for token, vec := range item.MultiVector {
			if len(vec) != space.Dimensions {
				t.Fatalf("document %d token %d width = %d, space declares %d",
					i, token, len(vec), space.Dimensions)
			}
		}
	}

	best := bestBy(t, len(docs.Data), func(i int) float64 {
		return maxSim(queries.Data[0].MultiVector, docs.Data[i].MultiVector)
	})
	if best != 1 {
		t.Errorf("MaxSim preferred document %d, want 1 (the paraphrase)", best)
	}
}

// The dense projection must behave exactly like any other Embedder, so an
// application can adopt this provider before it is ready to store sparse
// output.
func TestE2E_Representations_DeepInfraDenseEmbedderCompatibility(t *testing.T) {
	skipIfNoDeepInfraToken(t)
	ctx := representationCtx(t)

	encoder, err := deepinfra.New(deepinfra.BGEM3Model)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	checkEmbedder(t, ctx, encoder)
}

// ---------------------------------------------------------------------------
// Hugging Face
// ---------------------------------------------------------------------------

func skipIfNoHFToken(t *testing.T) {
	t.Helper()
	if os.Getenv("HF_TOKEN") == "" && os.Getenv("HUGGING_FACE_HUB_TOKEN") == "" {
		t.Skip("HF_TOKEN not set, skipping Hugging Face e2e test")
	}
}

// The shared router returns dense vectors and nothing else, so this proves
// dense retrieval and that the encoder refuses to claim more.
func TestE2E_Representations_HuggingFaceShared(t *testing.T) {
	skipIfNoHFToken(t)
	model := os.Getenv("HF_SHARED_MODEL")
	if model == "" {
		t.Skip("HF_SHARED_MODEL not set, skipping Hugging Face shared e2e test")
	}
	ctx := representationCtx(t)

	var opts []huggingface.SharedOption
	if query, document := os.Getenv("HF_QUERY_PROMPT"), os.Getenv("HF_DOCUMENT_PROMPT"); query != "" || document != "" {
		opts = append(opts, huggingface.WithPromptNames(query, document))
	}
	encoder, err := huggingface.NewShared(model, opts...)
	if err != nil {
		t.Fatalf("NewShared: %v", err)
	}

	if caps := encoder.Capabilities(); len(caps.Outputs) != 1 ||
		caps.Outputs[0] != agentic.RepresentationDense {
		t.Fatalf("shared router advertises %v, want dense only", caps.Outputs)
	}
	if _, err := agentic.EncodeDocuments(ctx, encoder, representationDocs,
		agentic.RepresentationSparse); !errors.Is(err, agentic.ErrUnsupportedRepresentation) {
		t.Fatalf("got %v, want the sparse request refused", err)
	}

	docs, err := encodeWithSupportedRole(ctx, encoder, representationDocs)
	if err != nil {
		t.Fatalf("encode documents: %v", err)
	}
	queries, err := encodeWithSupportedRole(ctx, encoder, []string{paraphraseQuery})
	if err != nil {
		t.Fatalf("encode query: %v", err)
	}

	space, _ := docs.Space(agentic.RepresentationDense)
	t.Logf("dense space %s", space)

	best := bestBy(t, len(docs.Data), func(i int) float64 {
		return cosine(queries.Data[0].Dense, docs.Data[i].Dense)
	})
	if best != budgetDoc {
		t.Errorf("nearest document = %d, want %d (the paraphrase)", best, budgetDoc)
	}
}

// encodeWithSupportedRole uses the document and query roles when the model has
// prompts configured for them, and the untyped form otherwise.
func encodeWithSupportedRole(
	ctx context.Context,
	encoder agentic.RepresentationEncoder,
	texts []string,
) (*agentic.RepresentationResponse, error) {
	inputType := agentic.EmbeddingInputNone
	if encoder.Capabilities().SupportsInputType(agentic.EmbeddingInputDocument) {
		inputType = agentic.EmbeddingInputDocument
	}
	return encoder.Encode(ctx, &agentic.RepresentationRequest{
		Input:     texts,
		InputType: inputType,
		Outputs:   []agentic.RepresentationKind{agentic.RepresentationDense},
	})
}

// A dedicated endpoint running the canonical handler is the path that carries
// sparse output, so this is where the same logical request is proven against a
// deployment you operate.
func TestE2E_Representations_HuggingFaceDedicated(t *testing.T) {
	skipIfNoHFToken(t)
	endpoint := os.Getenv("HF_ENDPOINT_URL")
	expectedSpace := os.Getenv("HF_DENSE_SPACE_ID")
	if endpoint == "" || expectedSpace == "" {
		t.Skip("HF_ENDPOINT_URL / HF_DENSE_SPACE_ID not set, skipping Hugging Face dedicated e2e test")
	}
	ctx := representationCtx(t)

	encoder, err := huggingface.NewDedicated(endpoint, huggingface.WithModel("BAAI/bge-m3"))
	if err != nil {
		t.Fatalf("NewDedicated: %v", err)
	}
	checkDedicatedEndpoint(t, ctx, encoder, expectedSpace)
}

// ---------------------------------------------------------------------------
// SageMaker
// ---------------------------------------------------------------------------

func TestE2E_Representations_SageMaker(t *testing.T) {
	endpoint := os.Getenv("SAGEMAKER_ENDPOINT")
	expectedSpace := os.Getenv("SAGEMAKER_DENSE_SPACE_ID")
	if endpoint == "" || expectedSpace == "" {
		t.Skip("SAGEMAKER_ENDPOINT / SAGEMAKER_DENSE_SPACE_ID not set, skipping SageMaker e2e test")
	}
	if os.Getenv("AWS_REGION") == "" && os.Getenv("AWS_DEFAULT_REGION") == "" {
		t.Skip("AWS_REGION not set, skipping SageMaker e2e test")
	}
	ctx := representationCtx(t)

	encoder, err := sagemaker.New(ctx, endpoint, sagemaker.WithModel("BAAI/bge-m3"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	checkDedicatedEndpoint(t, ctx, encoder, expectedSpace)
}

// checkDedicatedEndpoint runs the same proof against any endpoint speaking
// agentic.representations.v1, whichever transport carries it. That the two
// deployments are interchangeable here is the point of publishing a protocol.
func checkDedicatedEndpoint(
	t *testing.T,
	ctx context.Context,
	encoder agentic.RepresentationEncoder,
	expectedDenseSpaceID string,
) {
	t.Helper()

	docs, err := agentic.EncodeDocuments(ctx, encoder, representationDocs,
		agentic.RepresentationDense, agentic.RepresentationSparse)
	if err != nil {
		t.Fatalf("EncodeDocuments: %v", err)
	}
	if len(docs.Data) != len(representationDocs) {
		t.Fatalf("got %d representations for %d documents", len(docs.Data), len(representationDocs))
	}
	if docs.Usage.RequestCount != 1 {
		t.Errorf("request count = %d, want one call for both kinds", docs.Usage.RequestCount)
	}

	denseSpace, ok := docs.Space(agentic.RepresentationDense)
	if !ok {
		t.Fatal("response describes no dense space")
	}
	t.Logf("dense space %s", denseSpace)
	if denseSpace.ID != expectedDenseSpaceID {
		t.Fatalf("dense space ID = %q, want the configured %q; the deployment behind "+
			"this endpoint is not the one this index was built from",
			denseSpace.ID, expectedDenseSpaceID)
	}

	sparseSpace, ok := docs.Space(agentic.RepresentationSparse)
	if !ok {
		t.Fatal("response describes no sparse space")
	}
	if sparseSpace.Metric != agentic.SimilarityDotProduct {
		t.Errorf("sparse metric = %q, want dot_product", sparseSpace.Metric)
	}
	for i, item := range docs.Data {
		if len(item.Dense) != denseSpace.Dimensions {
			t.Fatalf("document %d dense width = %d", i, len(item.Dense))
		}
		if item.Sparse.Len() == 0 {
			t.Fatalf("document %d has no sparse coordinates", i)
		}
	}

	queries, err := agentic.EncodeQueries(ctx, encoder, []string{paraphraseQuery, rareTermQuery},
		agentic.RepresentationDense, agentic.RepresentationSparse)
	if err != nil {
		t.Fatalf("EncodeQueries: %v", err)
	}
	querySpace, _ := queries.Space(agentic.RepresentationDense)
	if querySpace.ID != denseSpace.ID {
		t.Fatalf("query space %s differs from document space %s", querySpace.ID, denseSpace.ID)
	}

	t.Run("dense finds the paraphrase", func(t *testing.T) {
		best := bestBy(t, len(docs.Data), func(i int) float64 {
			return cosine(queries.Data[0].Dense, docs.Data[i].Dense)
		})
		if best != budgetDoc {
			t.Errorf("nearest document = %d, want %d (the paraphrase)", best, budgetDoc)
		}
	})

	t.Run("sparse finds the rare term", func(t *testing.T) {
		best := bestBy(t, len(docs.Data), func(i int) float64 {
			return dotProduct(queries.Data[1].Sparse, docs.Data[i].Sparse)
		})
		if best != quenselDoc {
			t.Errorf("highest sparse score = %d, want %d (the rare term)", best, quenselDoc)
		}
	})
}

// ---------------------------------------------------------------------------
// Pinecone Inference
// ---------------------------------------------------------------------------

// The bound comes from the provider package, where it is recorded as a
// measurement rather than a specification. This test is what would catch it
// drifting: a live coordinate above the bound fails validation loudly instead
// of being stored out of range.

func TestE2E_Representations_PineconeSparse(t *testing.T) {
	apiKey := os.Getenv("PINECONE_API_KEY")
	model := os.Getenv("PINECONE_SPARSE_MODEL")
	if apiKey == "" || model == "" {
		t.Skip("PINECONE_API_KEY / PINECONE_SPARSE_MODEL not set, skipping Pinecone e2e test")
	}
	ctx := representationCtx(t)

	encoder, err := pinecone.New(model,
		pinecone.WithOutputs(agentic.RepresentationSparse),
		pinecone.WithSparseIndexSpace(pinecone.SparseEnglishIndexSpace),
		pinecone.WithReturnTokens(true),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	docs, _, err := encoder.EncodeWithTokens(ctx, &agentic.RepresentationRequest{
		Input:     representationDocs,
		InputType: agentic.EmbeddingInputDocument,
		Outputs:   []agentic.RepresentationKind{agentic.RepresentationSparse},
	})
	if err != nil {
		t.Fatalf("encode documents: %v", err)
	}

	space, _ := docs.Space(agentic.RepresentationSparse)
	t.Logf("sparse space %s", space)
	if space.Metric != agentic.SimilarityDotProduct {
		t.Errorf("sparse metric = %q, want dot_product", space.Metric)
	}
	if docs.Usage.RequestCount != 1 || docs.Usage.InputTokens < 0 {
		t.Errorf("usage = %+v", docs.Usage)
	}

	queries, queryTokens, err := encoder.EncodeWithTokens(ctx, &agentic.RepresentationRequest{
		Input:     []string{rareTermQuery, expansionQuery},
		InputType: agentic.EmbeddingInputQuery,
		Outputs:   []agentic.RepresentationKind{agentic.RepresentationSparse},
	})
	if err != nil {
		t.Fatalf("encode queries: %v", err)
	}

	// Query and passage roles encode into the same vocabulary, which is what
	// makes a query comparable to an indexed passage at all.
	querySpace, _ := queries.Space(agentic.RepresentationSparse)
	if querySpace.ID != space.ID {
		t.Fatalf("query space %s differs from document space %s", querySpace.ID, space.ID)
	}

	t.Run("sparse finds the rare term", func(t *testing.T) {
		best := bestBy(t, len(docs.Data), func(i int) float64 {
			return dotProduct(queries.Data[0].Sparse, docs.Data[i].Sparse)
		})
		if best != quenselDoc {
			t.Errorf("highest sparse score = %d, want %d (the rare term)", best, quenselDoc)
		}
	})

	// Whether a model weights terms the input never contained is a property to
	// measure, not to assume. pinecone-sparse-english-v0 does not, so this
	// records what it actually returned; a model that starts expanding shows
	// up in the log rather than silently changing what gets retrieved.
	t.Run("token diagnostics are aligned and reported", func(t *testing.T) {
		tokens := queryTokens[1]
		coordinates := queries.Data[1].Sparse.Len()
		if len(tokens) != coordinates {
			t.Fatalf("got %d tokens for %d coordinates", len(tokens), coordinates)
		}
		for i, token := range tokens {
			if token == "" {
				t.Errorf("coordinate %d has no token string", i)
			}
		}

		typed := map[string]bool{}
		for _, word := range strings.Fields(strings.ToLower(expansionQuery)) {
			typed[word] = true
		}
		var untyped []string
		for _, token := range tokens {
			if !typed[strings.ToLower(token)] {
				untyped = append(untyped, token)
			}
		}
		t.Logf("query %q -> %d coordinates %v", expansionQuery, coordinates, tokens)
		t.Logf("terms weighted that were not in the input: %v", untyped)
	})

	// The roles are not interchangeable: the query side weights every term at
	// 1.0, the passage side carries the saliency. Encoding documents with the
	// query role would erase the difference between a rare term and a
	// stopword, and nothing would report an error.
	t.Run("query and passage roles differ", func(t *testing.T) {
		asPassage, passageTokens, err := encoder.EncodeWithTokens(ctx, &agentic.RepresentationRequest{
			Input:     []string{rareTermQuery},
			InputType: agentic.EmbeddingInputDocument,
			Outputs:   []agentic.RepresentationKind{agentic.RepresentationSparse},
		})
		if err != nil {
			t.Fatalf("encode as passage: %v", err)
		}

		asQuery := queries.Data[0].Sparse
		passage := asPassage.Data[0].Sparse
		if asQuery.Len() != passage.Len() {
			t.Fatalf("roles produced %d and %d coordinates", asQuery.Len(), passage.Len())
		}

		identical := true
		for i := range asQuery.Values {
			if asQuery.Values[i] != passage.Values[i] {
				identical = false
			}
		}
		if identical {
			t.Error("query and passage weights are identical; the role is not reaching the API")
		}
		for i, token := range passageTokens[0] {
			t.Logf("  %-16q query %.4f   passage %.4f",
				token, asQuery.Values[i], passage.Values[i])
		}
	})
}
