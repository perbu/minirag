package minirag

import (
	"math"
	"sort"
)

// CosineSimilarity computes the cosine similarity between two vectors
// Returns a value between -1 and 1, where 1 means identical direction
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// Search performs similarity search on the vector index
// Returns top-k results sorted by similarity score (highest first)
func Search(index *VectorIndex, queryEmbedding []float32, topK int, threshold float32) []SearchResult {
	if len(queryEmbedding) != index.Dimension {
		return nil
	}

	results := make([]SearchResult, 0, len(index.Chunks))

	// Compute similarity for all chunks
	for i := range index.Chunks {
		score := CosineSimilarity(queryEmbedding, index.Embeddings[i])

		// Only include results above threshold
		if score >= threshold {
			results = append(results, SearchResult{
				Chunk: index.Chunks[i],
				Score: score,
			})
		}
	}

	// Sort by score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Return top-k results
	if topK > 0 && topK < len(results) {
		results = results[:topK]
	}

	return results
}

// LoadIndex creates a VectorIndex from EmbeddingData
func LoadIndex(data *EmbeddingData) *VectorIndex {
	return &VectorIndex{
		Chunks:     data.Chunks,
		Embeddings: data.Embeddings,
		Dimension:  data.Dimension,
	}
}
