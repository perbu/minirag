package embedder

import (
	"context"
	"errors"
	"math"
	"os"

	openai "github.com/sashabaranov/go-openai"
)

// OpenAIEmbedder uses OpenAI API for embeddings
type OpenAIEmbedder struct {
	client *openai.Client
	model  string
	dim    int
}

// NewOpenAIEmbedder creates an OpenAI embedder
func NewOpenAIEmbedder(model string) (*OpenAIEmbedder, error) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		return nil, errors.New("OPENAI_API_KEY environment variable not set")
	}

	client := openai.NewClient(key)

	// Set dimension based on model
	dim := 1536 // default for text-embedding-3-small
	if model == "text-embedding-3-large" {
		dim = 3072
	}

	return &OpenAIEmbedder{
		client: client,
		model:  model,
		dim:    dim,
	}, nil
}

// Embed generates an embedding for a single text
func (e *OpenAIEmbedder) Embed(text string) ([]float32, error) {
	// Validate input
	if len(text) == 0 {
		return nil, errors.New("cannot embed empty text")
	}

	ctx := context.Background()

	resp, err := e.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Model: openai.EmbeddingModel(e.model),
		Input: []string{text},
	})
	if err != nil {
		return nil, errors.New("OpenAI API error: " + err.Error())
	}

	if len(resp.Data) == 0 {
		return nil, errors.New("no embedding data returned from API")
	}

	// Convert float64 to float32
	v64 := resp.Data[0].Embedding
	v := make([]float32, len(v64))
	for i := range v64 {
		v[i] = float32(v64[i])
	}

	// L2 normalize (important for cosine similarity)
	l2normalize(v)

	return v, nil
}

// EmbedBatch generates embeddings for multiple texts with parallel processing
func (e *OpenAIEmbedder) EmbedBatch(texts []string) ([][]float32, error) {
	return e.EmbedBatchWithProgress(texts, nil)
}

// EmbedBatchWithProgress generates embeddings with optional progress callback
// progressFn is called with (completed, total) after each embedding
func (e *OpenAIEmbedder) EmbedBatchWithProgress(texts []string, progressFn func(int, int)) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	errChan := make(chan error, len(texts))
	sem := make(chan struct{}, 10) // Limit concurrent API calls to 10
	completed := make(chan int, len(texts))

	for i := range texts {
		sem <- struct{}{} // Acquire semaphore
		go func(idx int) {
			defer func() { <-sem }() // Release semaphore

			emb, err := e.Embed(texts[idx])
			if err != nil {
				errChan <- err
				completed <- 0
				return
			}
			embeddings[idx] = emb
			errChan <- nil
			completed <- 1
		}(i)
	}

	// Wait for all goroutines to complete and track progress
	count := 0
	for i := 0; i < len(texts); i++ {
		if err := <-errChan; err != nil {
			return nil, err
		}
		count += <-completed
		if progressFn != nil {
			progressFn(count, len(texts))
		}
	}

	return embeddings, nil
}

// Dimension returns the embedding dimension
func (e *OpenAIEmbedder) Dimension() int {
	return e.dim
}

// ModelInfo returns model information
func (e *OpenAIEmbedder) ModelInfo() string {
	return "openai-" + e.model
}

// l2normalize normalizes a vector to unit length
func l2normalize(v []float32) {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	if sum == 0 {
		return
	}
	inv := float32(1.0 / math.Sqrt(float64(sum)))
	for i := range v {
		v[i] *= inv
	}
}
