#!/bin/bash
# Quick smoke test for jina-reranker-v3
# Usage: bash tests/smoke_reranker.sh

set -e

BINARY="$(dirname "$0")/../build/bin/llama-embedding"
MODEL="/home/elias/projects/jina-test/jina-reranker-v3-projector.gguf"

echo "Binary: $BINARY"
echo "Model:  $MODEL"
echo ""

run_test() {
    local name="$1"
    local prompt="$2"
    echo "--- $name ---"
    echo "Prompt: $prompt"
    $BINARY -m "$MODEL" --pooling rank --prompt "$prompt" -o json 2>/dev/null | head -1
    echo ""
}

run_test "Relevant pair" "What is the capital of France?[/INST]Paris is the capital of France."
run_test "Irrelevant pair" "What is the capital of France?[/INST]Python is a programming language."
run_test "Short query" "AI[/INST]Artificial intelligence is transforming the world."
run_test "Long document" "What is climate change?[/INST]Climate change refers to long-term shifts in global temperatures and weather patterns."

echo "Smoke tests complete!"
