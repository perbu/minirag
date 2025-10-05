# MiniRAG

A Retrieval-Augmented Generation (RAG) library for Go that provides semantic search over markdown documentation.

## Features

- Three packages with minimal dependencies
- Semantic search using OpenAI embeddings
- Supports pre-built indexes or custom index creation
- In-memory cosine similarity search
- Can bundle documentation and embeddings in binaries

## Installation

```bash
go get github.com/perbu/minirag
```

## Quick Start

Basic usage:

### 1. Load an Index

```go
import "github.com/perbu/minirag/pkg/minirag"

index, err := minirag.LoadIndexFromFile("embeddings/index.gob")
if err != nil {
log.Fatal(err)
}
```

### 2. Embed Your Query

```go
import "github.com/perbu/minirag/pkg/embedder"

emb, _ := embedder.NewOpenAIEmbedder("text-embedding-3-small")
queryEmbedding, _ := emb.Embed("How do I configure authentication?")
```

### 3. Search

```go
results := minirag.Search(index, queryEmbedding, 5, 0.7)

for _, result := range results {
fmt.Printf("%s: %s\n", result.Chunk.Path, result.Chunk.Content)
}
```

## API Guide

### Complete Example: Search Pre-built Index

```go
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/perbu/minirag/pkg/embedder"
	"github.com/perbu/minirag/pkg/minirag"
)

func main() {
	// Load pre-built index
	index, err := minirag.LoadIndexFromFile("embeddings/index.gob")
	if err != nil {
		log.Fatal(err)
	}

	// Initialize embedder (requires OPENAI_API_KEY env var)
	emb, err := embedder.NewOpenAIEmbedder("text-embedding-3-small")
	if err != nil {
		log.Fatal(err)
	}

	// Embed query
	query := "How do I configure authentication?"
	queryEmbedding, err := emb.Embed(query)
	if err != nil {
		log.Fatal(err)
	}

	// Search: top 5 results, minimum similarity 0.7
	results := minirag.Search(index, queryEmbedding, 5, 0.7)

	// Display results
	for _, result := range results {
		fmt.Printf("Score: %.2f | %s", result.Score, result.Chunk.Path)
		if result.Chunk.Heading != "" {
			fmt.Printf(" [%s]", result.Chunk.Heading)
		}
		fmt.Printf("\n%s\n\n", result.Chunk.Content)
	}
}
```

### Building Your Own Index

```go
package main

import (
	"embed"
	"log"

	"github.com/perbu/minirag/pkg/embedder"
	"github.com/perbu/minirag/pkg/loader"
	"github.com/perbu/minirag/pkg/minirag"
)

//go:embed docs
var docsFS embed.FS

func main() {
	// 1. Load and chunk your documents
	chunks, _ := loader.LoadAndChunkAll(docsFS, "docs")

	// 2. Create embedder
	emb, _ := embedder.NewOpenAIEmbedder("text-embedding-3-small")

	// 3. Generate embeddings for all chunks
	texts := extractTexts(chunks)
	embeddings, _ := emb.EmbedBatch(texts)

	// 4. Build the index
	index := minirag.BuildIndex(chunks, embeddings, emb.Dimension())

	// 5. Save for later use
	minirag.SaveIndex(index, "my-index.gob")
}

func extractTexts(chunks []minirag.Chunk) []string {
	texts := make([]string, len(chunks))
	for i, chunk := range chunks {
		texts[i] = chunk.Content
	}
	return texts
}
```

### Testing Without OpenAI API

Use the simple embedder for local testing:

```go
// No API key needed
emb := embedder.NewSimpleEmbedder(128)

// Same interface as OpenAI embedder
embedding, _ := emb.Embed("test query")
embeddings, _ := emb.EmbedBatch([]string{"doc1", "doc2"})
```

### Custom Chunking

Process individual documents:

```go
content := `# Introduction
This is the intro.

# Getting Started
Here's how to start.`

chunks := loader.ChunkDocument("guide.md", content)

// Each chunk contains:
// - Path: "guide.md"
// - Heading: "Introduction" or "Getting Started"
// - Content: section text
// - Offset: position in original file
```

### Progress Tracking for Large Batches

```go
emb, _ := embedder.NewOpenAIEmbedder("text-embedding-3-small")
texts := []string{"doc1", "doc2", "doc3", /* ... */}

// Cast to access progress callback
openaiEmb := emb.(*embedder.OpenAIEmbedder)

embeddings, _ := openaiEmb.EmbedBatchWithProgress(texts,
func (completed, total int) {
fmt.Printf("Progress: %d/%d\n", completed, total)
},
)
```

## API Reference

### Package: `minirag`

Core types and search functions.

**Types:**

```go
type Chunk struct {
Path    string // e.g., "api/auth.md"
Content string // Section text
Heading string // e.g., "Authentication"
Offset  int // Position in file
}

type VectorIndex struct {
Chunks     []Chunk
Embeddings [][]float32
Dimension  int
}

type SearchResult struct {
Chunk Chunk
Score float32 // 0.0-1.0, higher = better match
}
```

**Functions:**

```go
// Load/save indexes
LoadIndexFromFile(path string) (*VectorIndex, error)
LoadIndexFromReader(r io.Reader) (*VectorIndex, error)
SaveIndex(index *VectorIndex, path string) error
BuildIndex(chunks []Chunk, embeddings [][]float32, dimension int) *VectorIndex

// Search
Search(index *VectorIndex, queryEmbedding []float32, topK int, threshold float32) []SearchResult
CosineSimilarity(a, b []float32) float32
```

### Package: `embedder`

Generate vector embeddings for text.

**Interface:**

```go
type Embedder interface {
Embed(text string) ([]float32, error)
EmbedBatch(texts []string) ([][]float32, error)
Dimension() int
ModelInfo() string
}
```

**Implementations:**

```go
// OpenAI embeddings (requires OPENAI_API_KEY)
NewOpenAIEmbedder(model string) (*OpenAIEmbedder, error)
// models: "text-embedding-3-small" (1536d), "text-embedding-3-large" (3072d)

// Simple hash-based embedder (for testing, no API needed)
NewSimpleEmbedder(dimension int) *SimpleEmbedder

// Progress tracking
(*OpenAIEmbedder).EmbedBatchWithProgress(
texts []string,
progressFn func (completed, total int),
) ([][]float32, error)
```

### Package: `loader`

Load and chunk markdown documents.

**Functions:**

```go
// Load from embedded filesystem
LoadAndChunkAll(fsys embed.FS, root string) ([]Chunk, error)
LoadDocuments(fsys embed.FS, root string) (map[string]string, error)

// Process individual documents
ChunkDocument(path, content string) []Chunk
```

Documents are automatically chunked by markdown headings (`#`, `##`, etc.).

## Configuration

### Environment Variables

```bash
# Required for OpenAI embeddings
export OPENAI_API_KEY=sk-...

# Or use .env file (automatically loaded)
echo "OPENAI_API_KEY=sk-..." > .env
```

### Models

| Model                    | Dimensions | Use Case                           |
|--------------------------|------------|------------------------------------|
| `text-embedding-3-small` | 1536       | Fast, cost-effective (recommended) |
| `text-embedding-3-large` | 3072       | Higher quality, more expensive     |

### Search Parameters

```go
results := minirag.Search(index, queryEmbedding, topK, threshold)
```

- **`topK`**: Max results to return (e.g., `5`, `10`)
- **`threshold`**: Minimum similarity score (0.0-1.0)
    - `0.5` - Very permissive
    - `0.7` - Balanced (recommended)
    - `0.8` - Strict, high precision

## Architecture

1. **Build time**:
    - Documents are chunked by markdown headings
    - Embeddings generated via OpenAI API
    - Index serialized to `.gob` file

2. **Runtime**:
    - Index loaded into memory
    - Queries embedded on-the-fly
    - Cosine similarity search (brute force, very fast for <100k chunks)

3. **Zero dependencies at runtime**:
    - No database needed
    - No external services (except for query embedding)
    - Can embed index in binary with `//go:embed`

## Best Practices

### Search Quality

- Markdown sections of 100-500 words produce better results
- Start with threshold 0.7, adjust based on precision/recall requirements
- `text-embedding-3-small` works for most use cases

### Performance

- Generate embeddings once, reuse the `.gob` file
- Use `EmbedBatch()` instead of loops (handles 10 concurrent requests)
- Index size is ~3KB per chunk; split large document sets into multiple indexes

### Cost

- Index generation: ~$0.02 per 1000 chunks (one-time)
- Per query: ~$0.00002 per query (embedding)
- Cache common queries or use simple embedder for tests

## Project Structure

```
minirag/
├── cmd/
│   ├── minirag/              # CLI search tool
│   ├── generate-embeddings/  # Build-time embedding generator
│   │   └── docs/            # Place your docs here
│   └── test-loader/          # Example document loader
├── pkg/                      # Public API - use these in your code
│   ├── minirag/             # Core types and search functions
│   ├── embedder/            # Embedding generation (OpenAI & simple)
│   └── loader/              # Document loading & chunking
└── embeddings/              # Generated index.gob (gitignored)
```

## Performance

Typical performance with OpenAI text-embedding-3-small:

- **Index generation**: ~10-20 chunks/second (API limited)
- **Search**: <100ms for 10k chunks (in-memory, single-threaded)
- **Binary size**: ~3KB per chunk + overhead
- **Memory usage**: ~4KB per chunk at runtime

Cost estimates:

- ~$0.02 per 1000 chunks (one-time)
- Queries: ~$0.00002 per query (if using OpenAI for query embedding)

## CLI Tools

Command-line tools included:

### Generate Embeddings

```bash
# 1. Add your docs
cp -r /path/to/docs cmd/generate-embeddings/docs/

# 2. Generate embeddings
export OPENAI_API_KEY=sk-...
go run cmd/generate-embeddings/main.go
# → Creates embeddings/index.gob
```

### Query from Command Line

```bash
# Build CLI
go build -o minirag cmd/minirag/main.go

# Search
./minirag "how to configure auth"
./minirag -top 10 -threshold 0.8 "authentication"
./minirag -full "API endpoints"
```

See `cmd/` directory for complete source code.

## Contributing

Contributions welcome! Please open an issue or PR.

## License

See LICENSE file for details.
