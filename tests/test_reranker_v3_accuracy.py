#!/usr/bin/env python3
"""
Comprehensive reranking accuracy test for jina-reranker-v3 in llama.cpp.

Tests that the model correctly ranks relevant documents higher than irrelevant
ones using the tab-separated query\tdocument format with --pooling rank.

Score strategies tested:
  A) Softmax probability of [yes] logit (first two dimensions)
  B) L2 norm of the full 512-dim embedding
  C) Mean of the embedding values
  D) [yes] logit raw value (first dimension)

Usage: python3 tests/test_reranker_v3_accuracy.py
"""

import subprocess
import numpy as np
import os
import sys
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BINARY = "/home/elias/projects/llama.cpp-jina-v3/build/bin/llama-embedding"
MODEL = "/home/elias/projects/jina-test/jina-reranker-v3-projector.gguf"
DIM = 512


# ---------------------------------------------------------------------------
# Core: call llama-embedding and parse output
# ---------------------------------------------------------------------------
def get_embedding(query: str, document: str) -> np.ndarray:
    """Run llama-embedding with RANK pooling, return the 512-dim vector."""
    prompt = f"{query}\t{document}"   # tab-separated triggers rerank template
    result = subprocess.run(
        [BINARY, "-m", MODEL, "--pooling", "rank", "--prompt", prompt],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-embedding failed (rc={result.returncode}):\n{result.stderr[:300]}"
        )

    values = []
    for line in result.stdout.splitlines():
        if "rerank score" in line and ":" in line:
            after_colon = line.split(":", 1)[1].strip()
            parts = after_colon.split()
            if parts:
                try:
                    values.append(float(parts[0]))
                except ValueError:
                    pass

    if len(values) != DIM:
        raise ValueError(
            f"Expected {DIM} values, got {len(values)}"
        )
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Scoring strategies
# ---------------------------------------------------------------------------
def score_yes_logit(emb):
    """Strategy D: raw [yes] logit (first dimension)."""
    return float(emb[0])


def score_yes_softmax(emb):
    """Strategy A: softmax probability of [yes] over [yes, [no]]."""
    yes, no = emb[0], emb[1]
    # numerical stability
    m = max(yes, no)
    exp_yes = np.exp(yes - m)
    exp_no = np.exp(no - m)
    return float(exp_yes / (exp_yes + exp_no))


def score_l2_norm(emb):
    """Strategy B: L2 norm of the full 512-dim embedding."""
    return float(np.linalg.norm(emb))


def score_mean(emb):
    """Strategy C: mean of the embedding values."""
    return float(np.mean(emb))


SCORERS = OrderedDict([
    ("yes_logit",    ("Raw [yes] logit",           score_yes_logit)),
    ("yes_softmax",  ("Softmax P([yes])",          score_yes_softmax)),
    ("l2_norm",      ("L2 norm of 512-dim vector", score_l2_norm)),
    ("mean",         ("Mean of embedding values",  score_mean)),
])


# ---------------------------------------------------------------------------
# Test cases: (name, query, [(label, document)])
# label: "RELEVANT" or "IRRELEVANT" (or "PARTIAL" for nuanced cases)
# ---------------------------------------------------------------------------
TEST_CASES = [
    # --- Basic relevant vs irrelevant ---
    {
        "name": "Capital of France",
        "query": "What is the capital of France?",
        "docs": [
            ("RELEVANT",    "Paris is the capital and largest city of France."),
            ("IRRELEVANT",  "The Python programming language was created by Guido van Rossum."),
            ("IRRELEVANT",  "The stock market reached an all-time high today."),
        ],
    },
    {
        "name": "Install Python on Ubuntu",
        "query": "How do I install Python on Ubuntu?",
        "docs": [
            ("RELEVANT",    "To install Python on Ubuntu, run: sudo apt install python3"),
            ("IRRELEVANT",  "The weather in Tokyo is usually mild during spring."),
            ("IRRELEVANT",  "Paris is known for its cuisine and art museums."),
        ],
    },
    {
        "name": "Climate Change Causes",
        "query": "What causes climate change?",
        "docs": [
            ("RELEVANT",    "Climate change is primarily caused by burning fossil fuels like coal, oil and gas."),
            ("IRRELEVANT",  "The best pizza place in New York is Joe's Pizza."),
            ("IRRELEVANT",  "Shakespeare wrote Hamlet in the early 1600s."),
        ],
    },
    {
        "name": "Photosynthesis",
        "query": "How does photosynthesis work?",
        "docs": [
            ("RELEVANT",    "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."),
            ("IRRELEVANT",  "The stock market crashed in 1929."),
            ("IRRELEVANT",  "Beyonce released her album Renaissance in 2022."),
        ],
    },
    {
        "name": "HTTP 404",
        "query": "What does HTTP 404 mean?",
        "docs": [
            ("RELEVANT",    "HTTP 404 means the requested resource was not found on the server."),
            ("IRRELEVANT",  "The speed of light is approximately 300,000 km/s."),
            ("IRRELEVANT",  "Bread is made from flour, water, yeast, and salt."),
        ],
    },
    # --- Multi-doc ranking: correct answer not first ---
    {
        "name": "Largest Planet (correct in middle)",
        "query": "What is the largest planet in our solar system?",
        "docs": [
            ("IRRELEVANT",  "The smallest country in the world is Vatican City."),
            ("RELEVANT",    "Jupiter is the largest planet in our solar system."),
            ("IRRELEVANT",  "The Pacific Ocean is the largest ocean on Earth."),
        ],
    },
    {
        "name": "1984 Author (correct last)",
        "query": "Who wrote the book 1984?",
        "docs": [
            ("IRRELEVANT",  "The Great Gatsby was written by F. Scott Fitzgerald."),
            ("IRRELEVANT",  "To Kill a Mockingbird was written by Harper Lee."),
            ("RELEVANT",    "1984 was written by George Orwell."),
        ],
    },
    # --- Two relevant docs: pick the best ---
    {
        "name": "Sort list in Python (best vs related)",
        "query": "How do I sort a list in Python?",
        "docs": [
            ("RELEVANT",    "Use the sorted() function or list.sort() method to sort a list in Python."),
            ("PARTIAL",     "Python lists can contain mixed data types."),
            ("IRRELEVANT",  "The capital of Brazil is Brasilia."),
        ],
    },
    # --- Harder: subtle distinction ---
    {
        "name": "Machine Learning vs AI",
        "query": "What is deep learning?",
        "docs": [
            ("RELEVANT",    "Deep learning is a subset of machine learning that uses neural networks with multiple layers."),
            ("PARTIAL",     "Artificial intelligence is the simulation of human intelligence by machines."),
            ("IRRELEVANT",  "The human body has 206 bones."),
        ],
    },
    {
        "name": "DNA and Genetics",
        "query": "What is DNA?",
        "docs": [
            ("RELEVANT",    "DNA is a molecule that carries genetic instructions for the development and function of living organisms."),
            ("IRRELEVANT",  "The Renaissance was a period of cultural revival in Europe."),
            ("IRRELEVANT",  "The Great Wall of China is over 13,000 miles long."),
        ],
    },
    # --- Edge cases ---
    {
        "name": "Short query",
        "query": "AI",
        "docs": [
            ("RELEVANT",    "Artificial intelligence is transforming technology and healthcare."),
            ("IRRELEVANT",  "The recipe calls for two cups of flour."),
        ],
    },
    {
        "name": "Long relevant document",
        "query": "What is quantum computing?",
        "docs": [
            ("RELEVANT",    "Quantum computing uses quantum mechanical phenomena such as superposition and entanglement to process information. Unlike classical computers that use bits, quantum computers use quantum bits or qubits. These can exist in multiple states simultaneously, allowing quantum computers to solve certain problems much faster than classical computers. Applications include cryptography, drug discovery, and optimization problems."),
            ("IRRELEVANT",  "The Mona Lisa was painted by Leonardo da Vinci."),
        ],
    },
]


# ---------------------------------------------------------------------------
# Scoring cache: avoid re-running the same query\tdoc pair
# ---------------------------------------------------------------------------
_embedding_cache = {}


def get_cached_embedding(query, document):
    key = (query, document)
    if key not in _embedding_cache:
        _embedding_cache[key] = get_embedding(query, document)
    return _embedding_cache[key]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def run_test_case(case, scorer_name, scorer_label, scorer_fn):
    """Run one test case with one scorer. Returns (passed, details)."""
    query = case["query"]
    docs = case["docs"]

    results = []
    for label, doc in docs:
        emb = get_cached_embedding(query, doc)
        score = scorer_fn(emb)
        results.append((label, doc, score))

    # Sort by score descending to get ranking
    ranked = sorted(results, key=lambda x: x[2], reverse=True)

    # Find the relevant doc's rank
    relevant_rank = None
    for i, (label, doc, score) in enumerate(ranked):
        if label == "RELEVANT":
            relevant_rank = i
            break

    # Determine pass/fail
    # For cases with PARTIAL: RELEVANT should be ranked above PARTIAL and IRRELEVANT
    has_partial = any(l == "PARTIAL" for l, _, _ in results)

    passed = False
    if relevant_rank is not None:
        if relevant_rank == 0:
            passed = True
        elif relevant_rank == 1 and has_partial:
            # Accept if a PARTIAL doc is ranked above RELEVANT (lenient)
            # but RELEVANT should beat IRRELEVANT
            for i, (label, doc, score) in enumerate(ranked):
                if label == "IRRELEVANT" and i < relevant_rank:
                    passed = False
                    break
            else:
                passed = True

    return passed, ranked, relevant_rank


def run_all_tests():
    """Run all test cases with all scoring strategies."""
    print("=" * 78)
    print("  jina-reranker-v3: Comprehensive Reranking Accuracy Test")
    print("=" * 78)
    print()
    print(f"  Binary: {BINARY}")
    print(f"  Model:  {MODEL}")
    print(f"  Test cases: {len(TEST_CASES)}")
    print(f"  Scoring strategies: {len(SCORERS)}")
    print()

    if not os.path.exists(BINARY):
        sys.exit(f"ERROR: Binary not found: {BINARY}")
    if not os.path.exists(MODEL):
        sys.exit(f"ERROR: Model not found: {MODEL}")

    # Track results per scorer
    all_results = {}
    for scorer_name in SCORERS:
        all_results[scorer_name] = {"passed": 0, "total": 0, "cases": []}

    # Phase 1: Fetch all embeddings (cached)
    print("Phase 1: Computing embeddings for all query/document pairs...")
    print("-" * 78)
    total_pairs = sum(len(c["docs"]) for c in TEST_CASES)
    print(f"  Total pairs: {total_pairs}")
    print()

    # Phase 2: Run scoring strategies
    for scorer_name, (scorer_label, scorer_fn) in SCORERS.items():
        print(f"\n{'=' * 78}")
        print(f"  Scoring Strategy: {scorer_label} ({scorer_name})")
        print(f"{'=' * 78}")

        for ci, case in enumerate(TEST_CASES):
            passed, ranked, relevant_rank = run_test_case(
                case, scorer_name, scorer_label, scorer_fn
            )

            all_results[scorer_name]["total"] += 1
            if passed:
                all_results[scorer_name]["passed"] += 1

            status = "PASS" if passed else "FAIL"
            print(f"\n  [{ci+1}/{len(TEST_CASES)}] {case['name']} [{status}]")
            print(f"    Query: {case['query']}")

            for ri, (label, doc, score) in enumerate(ranked):
                marker = " <-- RELEVANT" if label == "RELEVANT" else ""
                print(f"    #{ri+1} [{label}] score={score:.6f} | {doc[:60]}{marker}")

            if relevant_rank is not None:
                print(f"    Relevant doc rank: #{relevant_rank + 1}")

    # Phase 3: Summary
    print(f"\n{'=' * 78}")
    print("  SUMMARY")
    print(f"{'=' * 78}")

    best_scorer = None
    best_pass = -1

    for scorer_name, stats in all_results.items():
        label, _ = SCORERS[scorer_name]
        pct = 100.0 * stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        emoji = "✅" if pct == 100 else ("⚠️ " if pct >= 75 else "❌")
        print(f"\n  {emoji} {scorer_name:15s} ({label})")
        print(f"     Passed: {stats['passed']}/{stats['total']} ({pct:.0f}%)")
        if stats["passed"] > best_pass:
            best_pass = stats["passed"]
            best_scorer = scorer_name

    print(f"\n{'=' * 78}")
    best_label, _ = SCORERS[best_scorer]
    best_pct = 100.0 * all_results[best_scorer]["passed"] / all_results[best_scorer]["total"]
    print(f"  Best strategy: {best_scorer} ({best_label}) — {best_pct:.0f}%")

    if best_pct == 100:
        print("  ✅ ALL TESTS PASSED — PERFECT RERANKING")
    elif best_pct >= 75:
        print("  ⚠️  Mostly correct — some edge cases missed")
    else:
        print("  ❌ Poor accuracy — model or scoring strategy needs improvement")
    print(f"{'=' * 78}")

    return best_pct


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        pct = run_all_tests()
        sys.exit(0 if pct >= 75 else 1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)
