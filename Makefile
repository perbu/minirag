.PHONY: embeddings build clean test

# Generate embeddings (run this first, or when docs change)
embeddings: embeddings/index.gob
	mkdir -p cmd/minirag/embeddings
	cp embeddings/index.gob cmd/minirag/embeddings/

embeddings/index.gob:
	go run cmd/generate-embeddings/main.go

# Build the CLI binary
build:
	@if [ ! -f cmd/minirag/embeddings/index.gob ]; then \
		echo "Error: embeddings/index.gob not found in cmd/minirag/"; \
		echo "Run 'make embeddings' first to generate the index."; \
		exit 1; \
	fi
	go build -o minirag cmd/minirag/main.go

# Build both tools
all: embeddings build

# Clean build artifacts
clean:
	rm -f minirag generate-embeddings
	rm -rf embeddings/
	rm -rf cmd/minirag/embeddings/

# Run tests
test:
	go test ./...

# Quick test of the CLI
demo: build
	./minirag "configuration" --top 3
