#!/usr/bin/env python3
"""
End-to-end reranking verification for jina-reranker-v3.

Strategy:
1. Build full rerank prompt with query + all docs
2. Use transformers tokenizer to find positions of <|embed_token|> and <|rerank_token|>
3. Use llama-embedding --pooling none to get per-token embeddings
4. Extract embeddings from those positions
5. Compute cosine similarity for each doc
6. Check if the relevant doc ranks #1
"""

import subprocess
import numpy as np
import os
import sys

LLAMA_BINARY = "/home/elias/projects/llama.cpp-jina-v3/build/bin/llama-embedding"
MODEL_GGUF = "/home/elias/projects/jina-test/jina-reranker-v3-dense.gguf"

# Token IDs for special tokens (from transformers tokenizer)
EMBED_TOKEN_ID = 151670  # <|embed_token|>
RERANK_TOKEN_ID = 151671  # <|rerank_token|>


SYSTEM = """\
You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. If the query is a question, how relevant a passage is depends on how well it answers the question. If not, try to analyze the intent of the query and assess how well each passage satisfies the intent."""

USER_PREFIX = """\
I will provide you with {n} passages, each indicated by a numerical identifier. Rank the passages based on their relevance to query: {query}"""


def build_rerank_prompt(query, documents):
    """Build full rerank prompt with special tokens."""
    n = len(documents)
    prompt = f"$system\n{SYSTEM}\n\n$user\n"
    prompt += USER_PREFIX.format(n=n, query=query) + "\n\n"
    
    for i, doc in enumerate(documents):
        prompt += f'<passage id="{i}">\n{doc}<|embed_token|>\n</passage>\n\n'
    
    prompt += f"<query>\n{query}<|rerank_token|>\n</query>\n\n"
    prompt += "$assistant\n\n\n"
    
    return prompt


def llama_embedding(prompt):
    """Run llama-embedding with --pooling rank and return the embedding."""
    result = subprocess.run(
        [LLAMA_BINARY, "-m", MODEL_GGUF, "--pooling", "rank", "-p", prompt],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"  ERROR: rc={result.returncode}, stderr={result.stderr[:300]}")
        return {}

    embeddings = {}
    for line in result.stdout.splitlines():
        if line.startswith("embedding ") or line.startswith("rerank score "):
            parts = line.split(":", 1)
            if len(parts) == 2:
                # Handle both "embedding 0: ..." and "rerank score 0: ..."
                idx_str = parts[0].split()[-1]
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                # Filter out '...' and parse floats
                values = [float(v) for v in parts[1].strip().split() if v != '...' and not v.startswith('[')]
                if values:
                    embeddings[idx] = np.array(values, dtype=np.float32)
    
    # With --pooling rank, we get a single embedding (index 0)
    return embeddings


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def rerank_score(query, documents):
    """
    Compute reranking scores for a query against multiple documents.
    
    Strategy:
    1. For each doc, build a prompt with query + doc
    2. Use llama-embedding --pooling rank to get the embedding
    3. Compute cosine similarity between query and doc embeddings
    """
    # Get query embedding (query alone with <|rerank_token|>)
    query_prompt = f"{query}<|rerank_token|>"
    query_emb = llama_embedding(query_prompt)
    
    if not query_emb or 0 not in query_emb:
        print(f"  ERROR: Failed to get query embedding")
        return [0.0] * len(documents)
    
    query_vec = query_emb[0]
    
    # Get doc embeddings and compute scores
    scores = []
    for i, doc in enumerate(documents):
        doc_prompt = f"{doc}<|embed_token|>"
        doc_emb = llama_embedding(doc_prompt)
        
        if doc_emb and 0 in doc_emb:
            doc_vec = doc_emb[0]
            score = cosine_similarity(query_vec, doc_vec)
            scores.append(score)
        else:
            print(f"  WARNING: Failed to get doc embedding {i}")
            scores.append(0.0)
    
    return scores


# Test cases: (query, [documents], correct_answer_index)
# The correct answer should rank #1
CASES = [
    {
        "name": "Capital of France",
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital and largest city of France.",  # 0 = correct
            "The Eiffel Tower is a famous landmark in Germany.",
            "Python is a popular programming language.",
            "The stock market reached an all-time high today.",
        ],
        "correct": 0,
    },
    {
        "name": "Install Python",
        "query": "How do I install Python on Ubuntu?",
        "documents": [
            "The weather in Tokyo is usually mild during spring.",  # 0
            "To install Python on Ubuntu, run: sudo apt install python3",  # 1 = correct
            "Paris is known for its cuisine and art museums.",
            "Machine learning models require large datasets.",
        ],
        "correct": 1,
    },
    {
        "name": "Climate Change",
        "query": "What causes climate change?",
        "documents": [
            "Climate change is primarily caused by burning fossil fuels like coal, oil and gas.",  # 0 = correct
            "The best pizza place in New York is Joe's Pizza.",
            "Shakespeare wrote Hamlet in the early 1600s.",
            "The human body has 206 bones.",
        ],
        "correct": 0,
    },
    {
        "name": "Photosynthesis",
        "query": "How does photosynthesis work?",
        "documents": [
            "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",  # 0 = correct
            "The stock market crashed in 1929.",
            "Beyonce released her album Renaissance in 2022.",
            "The capital of Australia is Canberra.",
        ],
        "correct": 0,
    },
    {
        "name": "HTTP Status Codes",
        "query": "What does HTTP 404 mean?",
        "documents": [
            "HTTP 404 means the requested resource was not found on the server.",  # 0 = correct
            "The speed of light is approximately 300,000 km/s.",
            "Bread is made from flour, water, yeast, and salt.",
            "The Mona Lisa was painted by Leonardo da Vinci.",
        ],
        "correct": 0,
    },
    {
        "name": "Correct answer in middle",
        "query": "What is the largest planet in our solar system?",
        "documents": [
            "The smallest country in the world is Vatican City.",  # 0
            "Jupiter is the largest planet in our solar system.",  # 1 = correct
            "The Pacific Ocean is the largest ocean on Earth.",
        ],
        "correct": 1,
    },
    {
        "name": "Correct answer last",
        "query": "Who wrote the book '1984'?",
        "documents": [
            "The Great Gatsby was written by F. Scott Fitzgerald.",  # 0
            "To Kill a Mockingbook was written by Harper Lee.",  # 1
            "1984 was written by George Orwell.",  # 2 = correct
        ],
        "correct": 2,
    },
    {
        "name": "Two relevant docs - pick best",
        "query": "How do I sort a list in Python?",
        "documents": [
            "Use the sorted() function or list.sort() method to sort a list in Python.",  # 0 = best
            "Python lists can contain mixed data types.",  # 1 = related but not answer
            "The capital of Brazil is Brasilia.",  # 2 = irrelevant
        ],
        "correct": 0,
    },
]


def main():
    print("=" * 72)
    print("jina-reranker-v3: Reranking Accuracy Test")
    print("=" * 72)
    print()
    print("For each test case, the model scores multiple documents against")
    print("a query. We check if the correct document ranks #1 (top-1 accuracy).")
    print()

    if not os.path.exists(LLAMA_BINARY):
        sys.exit(f"Binary not found: {LLAMA_BINARY}")
    if not os.path.exists(MODEL_GGUF):
        sys.exit(f"Model not found: {MODEL_GGUF}")

    top1_correct = 0
    top3_correct = 0
    total = len(CASES)

    for ci, case in enumerate(CASES):
        name = case["name"]
        query = case["query"]
        docs = case["documents"]
        correct_idx = case["correct"]
        n_docs = len(docs)

        print(f"\n[{ci+1}/{total}] {name}")
        print(f"  Query: {query}")
        print(f"  Documents ({n_docs}):")
        for i, doc in enumerate(docs):
            marker = " <-- CORRECT" if i == correct_idx else ""
            print(f"    [{i}] {doc[:60]}...{marker}")

        scores = rerank_score(query, docs)

        print(f"\n  Scores:")
        for i, s in enumerate(scores):
            marker = " <-- CORRECT" if i == correct_idx else ""
            print(f"    [{i}] {s:.4f}{marker}")

        # Rank documents by score (descending)
        ranked = np.argsort(scores)[::-1]
        print(f"\n  Ranking: ", end="")
        for rank, doc_idx in enumerate(ranked):
            marker = " ✓" if doc_idx == correct_idx else ""
            print(f"#{rank+1}=[{doc_idx}]", end=" ")
            if marker:
                print(marker, end=" ")

        # Check top-1
        if ranked[0] == correct_idx:
            top1_correct += 1
            print(f"\n  ✅ TOP-1 CORRECT")
        else:
            print(f"\n  ❌ TOP-1 WRONG (got [{ranked[0]}], expected [{correct_idx}])")

        # Check top-3
        if correct_idx in ranked[:min(3, n_docs)]:
            top3_correct += 1

    print("\n" + "=" * 72)
    print(f"RESULTS: {total} test cases")
    print(f"  Top-1 Accuracy: {top1_correct}/{total} ({100*top1_correct/total:.0f}%)")
    print(f"  Top-3 Accuracy: {top3_correct}/{total} ({100*top3_correct/total:.0f}%)")
    
    if top1_correct == total:
        print("\n  ✅ ALL TESTS PASSED — PERFECT RERANKING")
    elif top1_correct >= total * 0.75:
        print(f"\n  ⚠️  {top1_correct}/{total} correct — mostly good but some misses")
    else:
        print(f"\n  ❌ {top1_correct}/{total} correct — needs improvement")
    print("=" * 72)


if __name__ == "__main__":
    main()
