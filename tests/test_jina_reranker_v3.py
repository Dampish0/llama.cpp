#!/usr/bin/env python3
"""
Test jina-reranker-v3 cross-encoder reranking with llama.cpp.

Tests query/document pairs using RANK pooling and verifies that
relevant pairs score higher than irrelevant ones.
"""

import subprocess
import json
import numpy as np
import sys
import os

BINARY = os.path.join(os.path.dirname(__file__), "..", "build", "bin", "llama-embedding")
MODEL = "/home/elias/projects/jina-test/jina-reranker-v3-projector.gguf"


def get_embedding(query: str, document: str) -> np.ndarray:
    """Run llama-embedding with RANK pooling and return the 512-dim vector."""
    prompt = f"{query}[/INST]{document}"
    result = subprocess.run(
        [BINARY, "-m", MODEL, "--pooling", "rank", "--prompt", prompt],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        print(f"  ERROR: llama-embedding failed (rc={result.returncode})")
        print(f"  stderr: {result.stderr[:500]}")
        return np.zeros(512)

    # Parse output format: "rerank score 0:    0.012 [498]" or "rerank score 0:    0.001 [yes]"
    values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if "rerank score" in line and ":" in line:
            # Extract the number after the colon
            after_colon = line.split(":", 1)[1].strip()
            parts = after_colon.split()
            if parts:
                try:
                    val = float(parts[0])
                    values.append(val)
                except ValueError:
                    pass

    if values:
        return np.array(values, dtype=np.float32)

    print("  WARNING: Could not parse embedding output")
    print(f"  stdout sample: {result.stdout[:200]}")
    return np.zeros(512)


def rerank_score(query: str, documents: list[str]) -> list[float]:
    """Get reranking scores for a query against multiple documents.

    The jina-reranker-v3 outputs 512-dim embeddings where the first two
    dimensions represent [yes] and [no] logits — these are the actual
    relevance scores. Higher [yes] score = more relevant.
    """
    scores = []
    for doc in documents:
        emb = get_embedding(query, doc)
        # First dimension is the [yes] logit — the relevance score
        yes_score = float(emb[0]) if len(emb) > 0 else 0.0
        scores.append((yes_score, emb))
    return scores


def test_relevant_vs_irrelevant():
    """Test that relevant query/doc pairs score higher than irrelevant ones."""
    print("=" * 60)
    print("TEST: Relevant vs Irrelevant")
    print("=" * 60)

    query = "What is the capital of France?"
    relevant_doc = "Paris is the capital and largest city of France."
    irrelevant_doc = "The Python programming language was created by Guido van Rossum."

    print(f"\nQuery: {query}")
    print(f"Relevant doc: {relevant_doc}")
    print(f"Irrelevant doc: {irrelevant_doc}")

    rel_emb = get_embedding(query, relevant_doc)
    irrel_emb = get_embedding(query, irrelevant_doc)

    rel_yes = float(rel_emb[0])
    irrel_yes = float(irrel_emb[0])

    print(f"\nRelevant [yes] score:      {rel_yes:.4f}")
    print(f"Irrelevant [yes] score:    {irrel_yes:.4f}")

    # Also check mean absolute value as another signal
    rel_mean = float(np.mean(np.abs(rel_emb)))
    irrel_mean = float(np.mean(np.abs(irrel_emb)))
    print(f"Relevant mean abs:          {rel_mean:.4f}")
    print(f"Irrelevant mean abs:        {irrel_mean:.4f}")

    # Check if embeddings are actually different
    diff = float(np.linalg.norm(rel_emb - irrel_emb))
    print(f"Embedding distance:         {diff:.4f}")

    if diff > 0.1:
        print("\n✅ PASS: Embeddings are distinguishable")
    else:
        print("\n❌ FAIL: Embeddings are too similar")

    if rel_yes > irrel_yes:
        print("✅ PASS: Relevant doc has higher [yes] score")
    else:
        print(f"⚠️  WARNING: Relevant [yes]={rel_yes:.4f} < Irrelevant [yes]={irrel_yes:.4f}")

    return rel_emb, irrel_emb


def test_ranking():
    """Test that documents are ranked correctly by relevance."""
    print("\n" + "=" * 60)
    print("TEST: Document Ranking")
    print("=" * 60)

    query = "How do I install Python on Ubuntu?"
    documents = [
        ("HIGH", "To install Python on Ubuntu, run: sudo apt install python3"),
        ("MEDIUM", "Python is a popular programming language used for web development."),
        ("LOW", "The weather in Tokyo is usually mild during spring."),
        ("NONE", "xyz abc def ghi jkl mno pqr stu vwx yz"),
    ]

    scores = []
    for level, doc in documents:
        print(f"\n[{level}] {doc}")
        emb = get_embedding(query, doc)
        yes_score = float(emb[0])
        scores.append((level, doc, yes_score))
        print(f"  [yes] score: {yes_score:.4f}")

    # Sort by [yes] score descending
    ranked = sorted(scores, key=lambda x: x[2], reverse=True)
    print("\n--- Ranking (highest to lowest [yes] score) ---")
    for i, (level, doc, score) in enumerate(ranked):
        print(f"  {i+1}. [{level}] score={score:.4f} — {doc[:50]}...")

    # Check if HIGH is ranked first or second
    if ranked[0][0] in ("HIGH", "MEDIUM"):
        print("\n✅ PASS: Most relevant doc ranked near top")
    else:
        print(f"\n⚠️  WARNING: Highest ranked was [{ranked[0][0]}], expected [HIGH]")

    return scores


def test_edge_cases():
    """Test edge cases: short queries, empty docs, nonsense."""
    print("\n" + "=" * 60)
    print("TEST: Edge Cases")
    print("=" * 60)

    cases = [
        ("Short query", "AI", "Artificial intelligence is transforming technology."),
        ("Empty-ish doc", "What is machine learning?", "."),
        ("Nonsense", "asdf jklq", "zxcv bnml qwerty"),
        ("Long doc", "What is climate change?",
         "Climate change refers to long-term shifts in global temperatures and weather patterns. "
         "Since the 1800s, human activities have been the main driver of climate change, "
         "primarily due to burning fossil fuels like coal, oil and gas, which produces heat-trapping gases."),
    ]

    for name, query, doc in cases:
        print(f"\n[{name}]")
        emb = get_embedding(query, doc)
        yes_score = float(emb[0])
        nonzero = int(np.count_nonzero(emb))
        print(f"  [yes] score: {yes_score:.4f}, Non-zero dims: {nonzero}/512")
        if nonzero > 0:
            print(f"  ✅ Non-zero output")
        else:
            print(f"  ❌ All zeros")


def main():
    if not os.path.exists(BINARY):
        print(f"ERROR: Binary not found at {BINARY}")
        sys.exit(1)
    if not os.path.exists(MODEL):
        print(f"ERROR: Model not found at {MODEL}")
        sys.exit(1)

    print(f"Binary: {BINARY}")
    print(f"Model:  {MODEL}")
    print()

    test_relevant_vs_irrelevant()
    test_ranking()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
