package loader

import (
	"bufio"
	"embed"
	"fmt"
	"io/fs"
	"path/filepath"
	"strings"

	"github.com/perbu/minirag/pkg/minirag"
)

// LoadDocuments reads all markdown files from the embedded filesystem
// and returns them as a slice of raw documents with paths
func LoadDocuments(fsys embed.FS, root string) (map[string]string, error) {
	docs := make(map[string]string)

	err := fs.WalkDir(fsys, root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Skip directories
		if d.IsDir() {
			return nil
		}

		// Only process markdown files
		if !strings.HasSuffix(path, ".md") {
			return nil
		}

		// Read file content
		content, err := fs.ReadFile(fsys, path)
		if err != nil {
			return fmt.Errorf("reading %s: %w", path, err)
		}

		// Store with path relative to root
		relPath, err := filepath.Rel(root, path)
		if err != nil {
			relPath = path
		}

		docs[relPath] = string(content)
		return nil
	})

	return docs, err
}

// ChunkDocument splits a document into semantic chunks based on markdown headings
func ChunkDocument(path, content string) []minirag.Chunk {
	var chunks []minirag.Chunk

	scanner := bufio.NewScanner(strings.NewReader(content))

	var currentHeading string
	var currentContent strings.Builder
	var currentOffset int
	lineOffset := 0

	flushChunk := func() {
		if currentContent.Len() > 0 {
			chunks = append(chunks, minirag.Chunk{
				Path:    path,
				Content: strings.TrimSpace(currentContent.String()),
				Heading: currentHeading,
				Offset:  currentOffset,
			})
		}
	}

	for scanner.Scan() {
		line := scanner.Text()

		// Check if this is a heading (starts with #)
		if strings.HasPrefix(line, "#") {
			// Flush previous chunk before starting new one
			flushChunk()

			// Start new chunk
			currentHeading = strings.TrimSpace(strings.TrimLeft(line, "#"))
			currentContent.Reset()
			currentOffset = lineOffset
		} else {
			// Add line to current chunk
			if currentContent.Len() > 0 {
				currentContent.WriteString("\n")
			}
			currentContent.WriteString(line)
		}

		lineOffset += len(line) + 1 // +1 for newline
	}

	// Flush final chunk
	flushChunk()

	// If no chunks were created (no headings), treat whole doc as one chunk
	if len(chunks) == 0 {
		chunks = append(chunks, minirag.Chunk{
			Path:    path,
			Content: strings.TrimSpace(content),
			Heading: "",
			Offset:  0,
		})
	}

	return chunks
}

// LoadAndChunkAll loads all documents and chunks them
func LoadAndChunkAll(fsys embed.FS, root string) ([]minirag.Chunk, error) {
	docs, err := LoadDocuments(fsys, root)
	if err != nil {
		return nil, err
	}

	var allChunks []minirag.Chunk
	for path, content := range docs {
		chunks := ChunkDocument(path, content)
		allChunks = append(allChunks, chunks...)
	}

	return allChunks, nil
}
