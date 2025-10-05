package main

import (
	_ "embed"
	"encoding/gob"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/joho/godotenv"
	"github.com/perbu/minirag/pkg/embedder"
	"github.com/perbu/minirag/pkg/minirag"
)

//go:embed embeddings/index.gob
var embeddedIndex []byte

func main() {
	// Load .env file if it exists (for API key)
	_ = godotenv.Load()

	// Parse command line flags
	top := flag.Int("top", 5, "number of results to return")
	threshold := flag.Float64("threshold", 0.0, "minimum similarity score")
	full := flag.Bool("full", false, "show full content instead of just paths")
	verbose := flag.Bool("verbose", false, "enable verbose output for debugging")
	context := flag.Int("context", 0, "number of surrounding chunks to show for context")
	flag.Parse()

	// Get query string
	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprintf(os.Stderr, "Usage: minirag [options] <query>\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	query := strings.Join(args, " ")

	// Step 1: Load embedded index
	if *verbose {
		fmt.Printf("[DEBUG] Loading embedded index (%d bytes)...\n", len(embeddedIndex))
	}

	var embData minirag.EmbeddingData
	decoder := gob.NewDecoder(strings.NewReader(string(embeddedIndex)))
	if err := decoder.Decode(&embData); err != nil {
		fmt.Fprintf(os.Stderr, "Error loading index: %v\n", err)
		os.Exit(1)
	}

	if *verbose {
		fmt.Printf("[DEBUG] Loaded %d chunks, %d embeddings (dim=%d, model=%s)\n",
			len(embData.Chunks), len(embData.Embeddings), embData.Dimension, embData.ModelInfo)
	}

	index := minirag.LoadIndex(&embData)

	// Step 2: Initialize embedder for query
	if os.Getenv("OPENAI_API_KEY") == "" {
		fmt.Fprintf(os.Stderr, "Error: OPENAI_API_KEY environment variable not set\n")
		fmt.Fprintf(os.Stderr, "Please set it in .env file or environment\n")
		os.Exit(1)
	}

	emb, err := embedder.NewOpenAIEmbedder("text-embedding-3-small")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing embedder: %v\n", err)
		os.Exit(1)
	}

	// Step 3: Embed query
	if *verbose {
		fmt.Printf("[DEBUG] Embedding query: %q\n", query)
	}

	queryEmbedding, err := emb.Embed(query)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error embedding query: %v\n", err)
		os.Exit(1)
	}

	if *verbose {
		fmt.Printf("[DEBUG] Query embedding dimension: %d\n", len(queryEmbedding))
	}

	// Step 4: Execute search
	if *verbose {
		fmt.Printf("[DEBUG] Searching with top=%d, threshold=%.2f\n", *top, *threshold)
	}

	results := minirag.Search(index, queryEmbedding, *top, float32(*threshold))

	if *verbose {
		fmt.Printf("[DEBUG] Found %d results\n\n", len(results))
	}

	// Step 5: Display results
	if len(results) == 0 {
		fmt.Println("No results found")
		return
	}

	fmt.Printf("Found %d results:\n\n", len(results))
	for i, result := range results {
		fmt.Printf("Score: %.2f | %s", result.Score, result.Chunk.Path)
		if result.Chunk.Heading != "" {
			fmt.Printf(" [%s]", result.Chunk.Heading)
		}
		fmt.Println()

		if *full || *context > 0 {
			fmt.Println()

			// Find surrounding chunks from the same file
			if *context > 0 {
				surroundingChunks := findSurroundingChunks(&embData, result.Chunk, *context)

				for j, chunk := range surroundingChunks {
					if chunk.Path == result.Chunk.Path && chunk.Offset == result.Chunk.Offset {
						fmt.Printf(">>> MATCHED CHUNK <<<\n")
					}
					if chunk.Heading != "" {
						fmt.Printf("[%s]\n", chunk.Heading)
					}
					fmt.Printf("%s\n", chunk.Content)
					if j < len(surroundingChunks)-1 {
						fmt.Println()
					}
				}
			} else if *full {
				fmt.Printf("%s\n", result.Chunk.Content)
			}

			if i < len(results)-1 {
				fmt.Println("\n" + strings.Repeat("-", 80) + "\n")
			}
		}
	}
}

// findSurroundingChunks returns chunks before and after the target chunk from the same file
func findSurroundingChunks(data *minirag.EmbeddingData, target minirag.Chunk, contextSize int) []minirag.Chunk {
	var result []minirag.Chunk
	targetIdx := -1

	// Find the target chunk index
	for i, chunk := range data.Chunks {
		if chunk.Path == target.Path && chunk.Offset == target.Offset {
			targetIdx = i
			break
		}
	}

	if targetIdx == -1 {
		return []minirag.Chunk{target}
	}

	// Collect chunks from the same file within context range
	start := targetIdx - contextSize
	if start < 0 {
		start = 0
	}
	end := targetIdx + contextSize + 1
	if end > len(data.Chunks) {
		end = len(data.Chunks)
	}

	for i := start; i < end; i++ {
		if data.Chunks[i].Path == target.Path {
			result = append(result, data.Chunks[i])
		}
	}

	return result
}
