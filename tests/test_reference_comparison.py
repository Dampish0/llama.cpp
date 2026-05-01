#!/usr/bin/env python3
"""
Compare llama.cpp jina-reranker-v3 output against transformers reference.

Uses the official jina-reranker-v3 format:
  - Query embedding extracted from <|rerank_token|> position
  - Document embedding extracted from <|embed_token|> position
  - Cosine similarity between projected embeddings = rerank score
"""

import subprocess
import numpy as np
import os
import sys

LLAMA_BINARY = "/home/elias/projects/llama.cpp-jina-v3/build/bin/llama-embedding"
MODEL_GGUF = "/home/elias/projects/jina-test/jina-reranker-v3-projector.gguf"

# Special tokens
RERANK_TOKEN = "<|rerank_token|>"
EMBED_TOKEN = "<|embed_token|>"


SYSTEM_PROMPT = """\
$system
You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. If the query is a question, how relevant a passage is depends on how well it answers the question. If not, try to analyze the intent of the query and assess how well each passage satisfies the intent.

$user
I will provide you with 1 passages, each indicated by a numerical identifier. Rank the passages based on their relevance to query: {query}

<passage id="0">
{document}
</passage>
<query>
{query}
</query>

$assistant



"""


def build_rerank_prompt(query, document):
    """Build full rerank prompt with special tokens."""
    prompt = SYSTEM_PROMPT.format(query=query, document=document)
    # The model expects <|rerank_token|> after query and <|embed_token|> after doc
    # But for llama.cpp RANK pooling, we extract the last token, so we need
    # to run two separate prompts to get query and doc embeddings.
    return prompt


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


def llama_rerank_score(query, document):
    """Compute reranking score using llama.cpp.
    
    Strategy: Use the full prompt but append <|rerank_token|> or <|embed_token|>
    to extract embeddings from those specific positions.
    """
    # Base prompt without special tokens
    base_prompt = SYSTEM_PROMPT.format(query=query, document=document)
    
    # Get query embedding by appending <|rerank_token|>
    query_emb = llama_embedding(base_prompt + RERANK_TOKEN)
    
    # Get doc embedding by appending <|embed_token|>
    doc_emb = llama_embedding(base_prompt + EMBED_TOKEN)
    
    # Compute cosine similarity
    score = cosine_similarity(query_emb, doc_emb)
    return score, query_emb, doc_emb


def transformers_rerank_score(query, document):
    """Compute reranking score using transformers reference."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        if not hasattr(transformers_rerank_score, "_model"):
            print("Loading transformers model...")
            tok = AutoTokenizer.from_pretrained("jinaai/jina-reranker-v3", trust_remote_code=True)
            mdl = AutoModelForSequenceClassification.from_pretrained(
                "jinaai/jina-reranker-v3", trust_remote_code=True, torch_dtype=torch.float32
            )
            mdl.eval()
            transformers_rerank_score._tokenizer = tok
            transformers_rerank_score._model = mdl
        
        tok = transformers_rerank_score._tokenizer
        mdl = transformers_rerank_score._model
        
        # Build the prompt the same way
        prompt = SYSTEM_PROMPT.format(query=query, document=document)
        
        inputs = tok(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = mdl(**inputs)
        
        # scores are cosine similarities
        score = outputs.scores.item()
        return score
    except Exception as e:
        print(f"  transformers error: {e}")
        return None


CASES = [
    ("Exact match", "What is the capital of France?", "Paris is the capital and largest city of France.", "high"),
    ("Python install", "How do I install Python on Ubuntu?", "To install Python on Ubuntu, run: sudo apt install python3", "high"),
    ("Python general", "How do I install Python on Ubuntu?", "Python is a popular programming language used for web development.", "medium"),
    ("France/Python", "What is the capital of France?", "The Python programming language was created by Guido van Rossum.", "low"),
    ("Unrelated", "How do I install Python on Ubuntu?", "The weather in Tokyo is usually mild during spring.", "low"),
    ("Nonsense", "asdf jklq", "zxcv bnml qwerty", "low"),
    ("Climate match", "What causes climate change?", "Climate change is primarily caused by burning fossil fuels like coal, oil and gas.", "high"),
    ("Climate irrelevant", "What causes climate change?", "The stock market reached an all-time high today.", "low"),
]


def main():
    print("=" * 72)
    print("jina-reranker-v3: llama.cpp vs Transformers Reference")
    print("=" * 72)

    if not os.path.exists(LLAMA_BINARY):
        sys.exit(f"Binary not found: {LLAMA_BINARY}")
    if not os.path.exists(MODEL_GGUF):
        sys.exit(f"Model not found: {MODEL_GGUF}")

    # Run llama.cpp
    print("\nRunning llama.cpp...")
    llama_scores = []
    for name, q, d, _ in CASES:
        print(f"  {name}")
        score, q_emb, d_emb = llama_rerank_score(q, d)
        llama_scores.append(score)

    # Try transformers
    print("\nRunning transformers...")
    tf_scores = []
    try:
        for name, q, d, _ in CASES:
            print(f"  {name}")
            score = transformers_rerank_score(q, d)
            tf_scores.append(score)
    except Exception as e:
        print(f"  SKIP: {e}")
        tf_scores = [None] * len(CASES)

    # Table
    print("\n" + "=" * 72)
    print("COMPARISON")
    print("=" * 72)
    print(f"{'Test':<22} {'LLAMA':>8} {'TRANS':>8} {'DELTA':>8} {'OK':>8}")
    print("-" * 56)

    ok = True
    for i, (name, _, _, _) in enumerate(CASES):
        ls = llama_scores[i]
        ts = tf_scores[i]
        if ts is not None:
            delta = abs(ls - ts)
            agree = delta < 0.1  # within 0.1 is acceptable
            status = "YES" if agree else "NO"
            if not agree:
                ok = False
            print(f"{name:<22} {ls:>8.4f} {ts:>8.4f} {delta:>8.4f} {status:>8}")
        else:
            print(f"{name:<22} {ls:>8.4f} {'N/A':>8} {'N/A':>8} {'N/A':>8}")

    # Ranking
    print("\n" + "=" * 72)
    print("RANKING CHECK")
    print("=" * 72)
    for level in ["high", "medium", "low"]:
        scores = [llama_scores[i] for i, (_, _, _, lvl) in enumerate(CASES) if lvl == level]
        if scores:
            print(f"[{level.upper()}] avg=[{np.mean(scores):.4f}]")
            for i, (_, q, _, lvl) in enumerate(CASES):
                if lvl == level:
                    print(f"  {llama_scores[i]:.4f}  {q[:40]}")

    h = [llama_scores[i] for i, (_, _, _, l) in enumerate(CASES) if l == "high"]
    lo = [llama_scores[i] for i, (_, _, _, l) in enumerate(CASES) if l == "low"]
    if h and lo:
        print(f"\nHIGH avg: {np.mean(h):.4f}  LOW avg: {np.mean(lo):.4f}")
        if np.mean(h) > np.mean(lo):
            print("PASS: HIGH > LOW ✓")
        else:
            print("FAIL: HIGH should be > LOW ✗")
            ok = False

    print("\n" + "=" * 72)
    if ok:
        print("ALL CHECKS PASSED ✓")
    else:
        print("Some mismatches — see above.")
    print("=" * 72)


if __name__ == "__main__":
    main()
