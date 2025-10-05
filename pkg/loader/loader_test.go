package loader

import (
	"embed"
	"testing"
)

//go:embed testdata/*
var testFS embed.FS

func TestChunkDocument(t *testing.T) {
	content := `# Introduction
This is the intro.

## Setup
How to set up.

Some more setup info.

## Usage
How to use it.
`

	chunks := ChunkDocument("test.md", content)

	if len(chunks) != 3 {
		t.Errorf("Expected 3 chunks, got %d", len(chunks))
	}

	// Check first chunk
	if chunks[0].Heading != "Introduction" {
		t.Errorf("Expected heading 'Introduction', got '%s'", chunks[0].Heading)
	}

	// Check second chunk
	if chunks[1].Heading != "Setup" {
		t.Errorf("Expected heading 'Setup', got '%s'", chunks[1].Heading)
	}

	// Check content
	if !contains(chunks[1].Content, "How to set up") {
		t.Errorf("Chunk content missing expected text")
	}
}

func TestChunkDocument_NoHeadings(t *testing.T) {
	content := "Just plain text with no headings."

	chunks := ChunkDocument("test.md", content)

	if len(chunks) != 1 {
		t.Errorf("Expected 1 chunk, got %d", len(chunks))
	}

	if chunks[0].Heading != "" {
		t.Errorf("Expected empty heading, got '%s'", chunks[0].Heading)
	}
}

func contains(s, substr string) bool {
	return len(s) > 0 && len(substr) > 0 && len(s) >= len(substr) && s[:len(substr)] == substr || len(s) > len(substr) && contains(s[1:], substr)
}
