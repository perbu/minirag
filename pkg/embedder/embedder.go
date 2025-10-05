package embedder

import (
	"fmt"
)

// Embedder interface for generating embeddings
type Embedder interface {
	Embed(text string) ([]float32, error)
	EmbedBatch(texts []string) ([][]float32, error)
	Dimension() int
	ModelInfo() string
}

// SimpleEmbedder is a placeholder implementation using basic hashing
// TODO: Replace with spago-based transformer embeddings
type SimpleEmbedder struct {
	dim int
}

// NewSimpleEmbedder creates a basic embedder (temporary until spago is integrated)
func NewSimpleEmbedder(dimension int) *SimpleEmbedder {
	return &SimpleEmbedder{dim: dimension}
}

// Embed generates a simple embedding vector from text
// This is a placeholder - will be replaced with actual transformer embeddings
func (e *SimpleEmbedder) Embed(text string) ([]float32, error) {
	// Simple word-based embedding for now
	// TODO: Use spago BERT/sentence transformers
	vec := make([]float32, e.dim)

	// Basic hash-based encoding (just for testing structure)
	for i, char := range text {
		idx := i % e.dim
		vec[idx] += float32(char) / 1000.0
	}

	// Normalize
	var norm float32
	for _, v := range vec {
		norm += v * v
	}
	if norm > 0 {
		norm = float32(1.0 / float32(norm))
		for i := range vec {
			vec[i] *= norm
		}
	}

	return vec, nil
}

// EmbedBatch generates embeddings for multiple texts
func (e *SimpleEmbedder) EmbedBatch(texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		emb, err := e.Embed(text)
		if err != nil {
			return nil, fmt.Errorf("embedding text %d: %w", i, err)
		}
		embeddings[i] = emb
	}
	return embeddings, nil
}

// Dimension returns the embedding dimension
func (e *SimpleEmbedder) Dimension() int {
	return e.dim
}

// ModelInfo returns model information
func (e *SimpleEmbedder) ModelInfo() string {
	return "simple-embedder-v1"
}
