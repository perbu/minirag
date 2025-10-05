package main

import (
	"embed"
	"encoding/gob"
	"fmt"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/joho/godotenv"
	"github.com/perbu/minirag/pkg/embedder"
	"github.com/perbu/minirag/pkg/loader"
	"github.com/perbu/minirag/pkg/minirag"
)

//go:embed all:docs
var docsFS embed.FS

const checkpointPath = "embeddings/checkpoint.gob"

type checkpoint struct {
	Chunks     []minirag.Chunk
	Embeddings [][]float32
	Completed  map[int]bool // Track which chunks are done
	ModelInfo  string
	Dimension  int
}

func loadCheckpoint() (*checkpoint, error) {
	file, err := os.Open(checkpointPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // No checkpoint exists
		}
		return nil, err
	}
	defer file.Close()

	var cp checkpoint
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&cp); err != nil {
		return nil, err
	}

	return &cp, nil
}

func saveCheckpoint(cp *checkpoint) error {
	if err := os.MkdirAll("embeddings", 0755); err != nil {
		return err
	}

	file, err := os.Create(checkpointPath + ".tmp")
	if err != nil {
		return err
	}

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(cp); err != nil {
		file.Close()
		return err
	}

	if err := file.Close(); err != nil {
		return err
	}

	// Atomic rename
	return os.Rename(checkpointPath+".tmp", checkpointPath)
}

func main() {
	// Load .env file if it exists
	_ = godotenv.Load()

	fmt.Println("MiniRAG Embedding Generation Tool")
	fmt.Println("==================================")
	fmt.Println()

	// Setup signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	var cpMutex sync.Mutex
	var currentCP *checkpoint

	go func() {
		<-sigChan
		fmt.Println("\n\n⚠ Interrupt received, saving checkpoint...")
		cpMutex.Lock()
		if currentCP != nil {
			if err := saveCheckpoint(currentCP); err != nil {
				fmt.Fprintf(os.Stderr, "Error saving checkpoint: %v\n", err)
				os.Exit(1)
			}
			fmt.Println("✓ Checkpoint saved. Run again to resume.")
		}
		cpMutex.Unlock()
		os.Exit(0)
	}()

	// Verify API key
	if os.Getenv("OPENAI_API_KEY") == "" {
		fmt.Fprintf(os.Stderr, "Error: OPENAI_API_KEY environment variable not set\n")
		fmt.Fprintf(os.Stderr, "Please set it in .env file or environment\n")
		os.Exit(1)
	}

	// Step 1: Load and chunk documents
	fmt.Println("Step 1: Loading and chunking documents...")
	chunks, err := loader.LoadAndChunkAll(docsFS, "docs")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading documents: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  ✓ Loaded %d chunks from documents\n\n", len(chunks))

	// Step 2: Initialize OpenAI embedder
	fmt.Println("Step 2: Initializing OpenAI embedder...")
	model := "text-embedding-3-small"
	emb, err := embedder.NewOpenAIEmbedder(model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing embedder: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  ✓ Embedder initialized (model=%s, dim=%d)\n\n", model, emb.Dimension())

	// Step 2.5: Check for existing checkpoint
	var cp *checkpoint
	existingCP, err := loadCheckpoint()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: Error loading checkpoint: %v\n", err)
		fmt.Println("Starting from scratch...")
	} else if existingCP != nil {
		completed := 0
		for _, done := range existingCP.Completed {
			if done {
				completed++
			}
		}
		fmt.Printf("Found checkpoint: %d/%d embeddings already generated\n", completed, len(chunks))

		// Verify checkpoint matches current docs
		if len(existingCP.Chunks) != len(chunks) || existingCP.ModelInfo != emb.ModelInfo() {
			fmt.Println("  ⚠ Checkpoint doesn't match current documents/model, starting fresh")
			cp = nil
		} else {
			cp = existingCP
			fmt.Println("  ✓ Resuming from checkpoint")
		}
	}

	// Initialize new checkpoint if needed
	if cp == nil {
		cp = &checkpoint{
			Chunks:     chunks,
			Embeddings: make([][]float32, len(chunks)),
			Completed:  make(map[int]bool),
			ModelInfo:  emb.ModelInfo(),
			Dimension:  emb.Dimension(),
		}
	}

	// Make checkpoint available to signal handler
	cpMutex.Lock()
	currentCP = cp
	cpMutex.Unlock()

	// Step 3: Generate embeddings with progress and checkpointing
	fmt.Println("Step 3: Generating embeddings...")

	// Count how many we need to do
	remaining := 0
	for i := range chunks {
		if !cp.Completed[i] {
			remaining++
		}
	}

	if remaining == 0 {
		fmt.Println("  ✓ All embeddings already generated!")
	} else {
		fmt.Printf("  (This will call OpenAI API %d times, ~$%.2f estimated cost)\n", remaining, float64(remaining)*0.00002)
		fmt.Printf("  Using parallel processing (up to 10 concurrent requests)...\n")

		var mu sync.Mutex
		completed := len(chunks) - remaining
		saveCounter := 0

		// Build list of indices to process (to avoid concurrent map read)
		toProcess := make([]int, 0, remaining)
		for i := range chunks {
			if !cp.Completed[i] {
				toProcess = append(toProcess, i)
			}
		}

		// Generate only missing embeddings
		var wg sync.WaitGroup
		errChan := make(chan error, len(toProcess))
		sem := make(chan struct{}, 10) // Limit concurrent requests

		for _, idx := range toProcess {
			wg.Add(1)
			sem <- struct{}{}
			go func(idx int) {
				defer wg.Done()
				defer func() { <-sem }()

				emb_vec, err := emb.Embed(chunks[idx].Content)
				if err != nil {
					errChan <- fmt.Errorf("chunk %d (%s): %w", idx, chunks[idx].Path, err)
					return
				}

				mu.Lock()
				cp.Embeddings[idx] = emb_vec
				cp.Completed[idx] = true
				completed++
				saveCounter++

				// Show progress
				if completed%10 == 0 || completed == len(chunks) {
					fmt.Printf("\r  Progress: %d/%d (%.1f%%)", completed, len(chunks), float64(completed)/float64(len(chunks))*100)
					if completed == len(chunks) {
						fmt.Println()
					}
				}

				// Save checkpoint every 50 embeddings
				if saveCounter >= 50 {
					saveCounter = 0
					if err := saveCheckpoint(cp); err != nil {
						fmt.Fprintf(os.Stderr, "\nWarning: Failed to save checkpoint: %v\n", err)
					}
				}
				mu.Unlock()

				errChan <- nil
			}(idx)
		}

		wg.Wait()
		close(errChan)

		// Check for errors - collect all errors first
		var embedErrors []error
		for err := range errChan {
			if err != nil {
				embedErrors = append(embedErrors, err)
			}
		}

		if len(embedErrors) > 0 {
			fmt.Fprintf(os.Stderr, "\n\n⚠ Encountered %d error(s) during embedding:\n", len(embedErrors))
			for _, err := range embedErrors {
				fmt.Fprintf(os.Stderr, "  - %v\n", err)
			}
			fmt.Println("\nProgress saved to checkpoint. Run again to resume.")
			mu.Lock()
			if saveErr := saveCheckpoint(cp); saveErr != nil {
				fmt.Fprintf(os.Stderr, "Error saving checkpoint: %v\n", saveErr)
			}
			mu.Unlock()
			os.Exit(1)
		}

		// Final checkpoint save
		if err := saveCheckpoint(cp); err != nil {
			fmt.Fprintf(os.Stderr, "Error saving final checkpoint: %v\n", err)
		}

		fmt.Printf("  ✓ Generated %d embeddings\n\n", len(chunks))
	}

	// Step 4: Create embedding data structure from checkpoint
	embData := minirag.EmbeddingData{
		Chunks:     cp.Chunks,
		Embeddings: cp.Embeddings,
		ModelInfo:  cp.ModelInfo,
		Dimension:  cp.Dimension,
	}

	// Step 5: Save to final index file
	fmt.Println("Step 4: Saving final index...")
	outputPath := "embeddings/index.gob"

	// Ensure directory exists
	if err := os.MkdirAll("embeddings", 0755); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating embeddings directory: %v\n", err)
		os.Exit(1)
	}

	file, err := os.Create(outputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output file: %v\n", err)
		os.Exit(1)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(embData); err != nil {
		fmt.Fprintf(os.Stderr, "Error encoding embeddings: %v\n", err)
		os.Exit(1)
	}

	// Get file size
	info, _ := file.Stat()
	sizeMB := float64(info.Size()) / (1024 * 1024)

	fmt.Printf("  ✓ Saved to %s (%.2f MB)\n\n", outputPath, sizeMB)

	// Clean up checkpoint file
	if err := os.Remove(checkpointPath); err != nil && !os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Warning: Could not remove checkpoint file: %v\n", err)
	}

	fmt.Println("Done! Embeddings are ready for use.")
	fmt.Println("Run 'make build' to create the CLI binary.")
}
