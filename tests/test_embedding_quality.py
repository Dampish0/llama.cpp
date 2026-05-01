#!/usr/bin/env python3
"""
Verify jina-reranker-v3 RANK pooling produces sensible embeddings.

Tests that:
1. Different prompts produce different embeddings (not all identical)
2. Similar query/doc pairs have higher cosine similarity than dissimilar ones
"""

import subprocess
import numpy as np
import os
import sys

LLAMA_BINARY = "/home/elias/projects/llama.cpp-jina-v3/build/bin/llama-embedding"
MODEL_GGUF = "/home/elias/projects/jina-test/jina-reranker-v3-projector.gguf"


def llama_embedding(prompt):
    """Run llama-embedding and return the 512-dim projected embedding."""
    result = subprocess.run(
        [LLAMA_BINARY, "-m", MODEL_GGUF, "--pooling", "rank", "--prompt", prompt],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"  ERROR: rc={result.returncode}, stderr={result.stderr[:300]}")
        return np.zeros(512)

    values = []
    for line in result.stdout.splitlines():
        if "rerank score" in line and ":" in line:
            after = line.split(":", 1)[1].strip().split()
            if after:
                try:
                    values.append(float(after[0]))
                except ValueError:
                    pass

    if len(values) >= 512:
        return np.array(values[:512], dtype=np.float32)
    elif len(values) > 0:
        arr = np.array(values, dtype=np.float32)
        arr = np.pad(arr, (0, 512 - len(arr)))
        return arr
    return np.zeros(512)


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def main():
    print("=" * 72)
    print("jina-reranker-v3: Embedding Verification")
    print("=" * 72)

    if not os.path.exists(LLAMA_BINARY):
        sys.exit(f"Binary not found: {LLAMA_BINARY}")
    if not os.path.exists(MODEL_GGUF):
        sys.exit(f"Model not found: {MODEL_GGUF}")

    # Test 1: Different prompts should produce different embeddings
    print("\nTest 1: Embeddings vary across different prompts")
    print("-" * 50)
    
    prompts = [
        "What is the capital of France?",
        "How do I install Python on Ubuntu?",
        "The weather is nice today.",
        "Machine learning is a subset of AI.",
    ]
    
    embeddings = []
    for p in prompts:
        emb = llama_embedding(p)
        embeddings.append(emb)
        print(f"  '{p[:40]}...' -> norm={np.linalg.norm(emb):.4f}")
    
    # Check pairwise similarity
    print("\n  Pairwise cosine similarity:")
    for i in range(len(prompts)):
        for j in range(i+1, len(prompts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"    [{i}] vs [{j}]: {sim:.4f}")
    
    # Test 2: Same prompt should produce same embedding (determinism)
    print("\nTest 2: Determinism (same prompt = same embedding)")
    print("-" * 50)
    
    emb1 = llama_embedding("What is AI?")
    emb2 = llama_embedding("What is AI?")
    diff = np.abs(emb1 - emb2).max()
    print(f"  Max absolute difference: {diff:.8f}")
    if diff < 1e-5:
        print("  PASS: Embeddings are deterministic ✓")
    else:
        print("  FAIL: Embeddings differ ✗")
    
    # Test 3: Similar prompts should have higher similarity
    print("\nTest 3: Similar vs dissimilar prompts")
    print("-" * 50)
    
    # Similar pair
    emb_france = llama_embedding("What is the capital of France?")
    emb_paris = llama_embedding("Paris is the capital of France.")
    sim_france_paris = cosine_similarity(emb_france, emb_paris)
    print(f"  France vs Paris: {sim_france_paris:.4f}")
    
    # Dissimilar pair
    emb_python = llama_embedding("How do I install Python?")
    emb_weather = llama_embedding("The weather in Tokyo is mild.")
    sim_python_weather = cosine_similarity(emb_python, emb_weather)
    print(f"  Python vs Weather: {sim_python_weather:.4f}")
    
    if sim_france_paris > sim_python_weather:
        print("  PASS: Similar > Dissimilar ✓")
    else:
        print("  FAIL: Similar should be > Dissimilar ✗")
    
    # Test 4: Projector output should be non-zero and reasonable
    print("\nTest 4: Projector output quality")
    print("-" * 50)
    
    emb = llama_embedding("What is AI?")
    print(f"  Norm: {np.linalg.norm(emb):.4f}")
    print(f"  Min: {emb.min():.6f}")
    print(f"  Max: {emb.max():.6f}")
    print(f"  Mean: {emb.mean():.6f}")
    print(f"  Std: {emb.std():.6f}")
    print(f"  Non-zero elements: {(emb != 0).sum()}")
    
    if np.linalg.norm(emb) > 0.1 and emb.std() > 0.01:
        print("  PASS: Reasonable output ✓")
    else:
        print("  FAIL: Output looks suspicious ✗")
    
    print("\n" + "=" * 72)
    print("Verification complete")
    print("=" * 72)


if __name__ == "__main__":
    main()
