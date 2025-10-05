package minirag

// Chunk represents a piece of a document with its content and metadata
type Chunk struct {
	Path    string // File path relative to docs/
	Content string // The actual text content
	Heading string // Section heading if applicable
	Offset  int    // Character offset in original file
}

// EmbeddingData holds all pre-computed embeddings and their associated chunks
type EmbeddingData struct {
	Chunks     []Chunk     // Document chunks
	Embeddings [][]float32 // Corresponding embeddings (same order as Chunks)
	ModelInfo  string      // Model name/version used
	Dimension  int         // Embedding vector dimension
}

// SearchResult represents a single search result with score
type SearchResult struct {
	Chunk Chunk
	Score float32
}

// VectorIndex holds the in-memory vector index for similarity search
type VectorIndex struct {
	Chunks     []Chunk     // Document chunks
	Embeddings [][]float32 // Corresponding embeddings (chunk[i] ↔ embedding[i])
	Dimension  int         // Embedding vector dimension
}
