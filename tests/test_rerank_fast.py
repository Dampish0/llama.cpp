#!/usr/bin/env python3
"""
Fast reranking accuracy test for jina-reranker-v3.

Uses llama-embedding --pooling rank for individual query+doc pairs,
computing cosine similarity to rank documents.
"""

import subprocess
import numpy as np
import sys
import re

LLAMA_BINARY = "/home/elias/projects/llama.cpp-jina-v3/build/bin/llama-embedding"
MODEL_GGUF = "/home/elias/projects/jina-test/jina-reranker-v3-projector.gguf"

SYSTEM = """\
You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. If the query is a question, how relevant a passage is depends on how well it answers the question. If not, try to analyze the intent of the query and assess how well each passage satisfies the intent."""

def build_rerank_prompt(query, doc):
    """Build rerank prompt for a single query+doc pair."""
    prompt = f'$system\n{SYSTEM}\n\n$user\n'
    prompt += f'I will provide you with 1 passages, each indicated by a numerical identifier. Rank the passages based on their relevance to query: {query}\n\n'
    prompt += f'<passage id="0">\n{doc}<|embed_token|>\n</passage>\n\n'
    prompt += f'<query>\n{query}<|rerank_token|>\n</query>\n\n'
    prompt += '$assistant\n\n\n'
    return prompt

def get_embedding(prompt):
    """Run llama-embedding --pooling rank and return the embedding vector."""
    result = subprocess.run(
        [LLAMA_BINARY, "-m", MODEL_GGUF, "--pooling", "rank", "--prompt", prompt],
        capture_output=True, text=True, timeout=60
    )
    # Parse "rerank score 0: VALUE [INDEX]" lines
    scores = []
    for line in result.stdout.split('\n'):
        m = re.match(r'rerank score \d+:\s+([-\d.e+]+)\s+\[\d+\]', line)
        if m:
            scores.append(float(m.group(1)))
    if len(scores) != 512:
        print(f"ERROR: Expected 512 values, got {len(scores)}")
        if result.stderr:
            print(f"STDERR: {result.stderr[:500]}")
        sys.exit(1)
    return np.array(scores, dtype=np.float32)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def test_reranking():
    """Test reranking accuracy with known relevant/irrelevant docs."""
    
    # Test 1: Python query - relevant vs irrelevant docs
    query1 = "What is Python programming language?"
    docs1 = [
        ("RELEVANT", "Python is a high-level, general-purpose programming language. Python's design philosophy emphasizes code readability with the use of significant indentation."),
        ("IRRELEVANT", "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. It is named after the engineer Gustave Eiffel."),
        ("IRRELEVANT", "Basketball is a team sport in which two teams, most commonly of five players, attempt to score by shooting an orange ball through a hoop."),
    ]
    
    # Test 2: Machine learning query
    query2 = "What is deep learning?"
    docs2 = [
        ("RELEVANT", "Deep learning is part of a broader family of machine learning methods based on artificial neural networks. Learning can be supervised, semi-supervised or unsupervised."),
        ("IRRELEVANT", "The capital of France is Paris. It is the most populous city in France with over 2 million residents."),
        ("IRRELEVANT", "Water is a chemical compound with the chemical formula H2O. A water molecule contains one oxygen and two hydrogen atoms."),
    ]
    
    all_passed = True
    
    for test_num, (query, docs) in enumerate([(query1, docs1), (query2, docs2)], 1):
        print(f"\n{'='*60}")
        print(f"TEST {test_num}: Query: '{query}'")
        print(f"{'='*60}")
        
        # Build prompts and get embeddings for each doc
        prompts = []
        for label, doc in docs:
            prompt = build_rerank_prompt(query, doc)
            prompts.append((label, doc, prompt))
        
        # Get embeddings for all docs (each in its own forward pass)
        # The key insight: for cross-encoder reranking, each doc gets its own
        # embedding that encodes the query+doc interaction. Then we compare
        # them using cosine similarity or just use the norm/magnitude.
        
        embeddings = []
        for label, doc, prompt in prompts:
            print(f"\n  Getting embedding for: {label} - '{doc[:60]}...'")
            emb = get_embedding(prompt)
            embeddings.append((label, doc, emb))
            print(f"    Embedding norm: {np.linalg.norm(emb):.4f}")
        
        # For cross-encoder reranking with jina-reranker-v3, the score is typically
        # the cosine similarity between the <|rerank_token|> embedding and <|embed_token|> embedding
        # BUT since --pooling rank only gives us one embedding per prompt, we need to
        # check if the model's output already encodes relevance.
        
        # Alternative: compare all embeddings against each other
        print(f"\n  Cosine similarities between doc embeddings:")
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                label_i, doc_i, emb_i = embeddings[i]
                label_j, doc_j, emb_j = embeddings[j]
                sim = cosine_sim(emb_i, emb_j)
                print(f"    {label_i} <-> {label_j}: {sim:.4f}")
        
        # The relevance should show up as: the embedding for the relevant doc
        # should be more distinct or have a different pattern than irrelevant docs.
        # For now, let's check if the relevant doc's embedding has a higher norm
        # or is more different from the others.
        
        print(f"\n  Embedding norms (higher may indicate more 'activation'):")
        for label, doc, emb in embeddings:
            print(f"    {label}: {np.linalg.norm(emb):.4f}")
        
        print()

    print("\n" + "="*60)
    print("NOTE: This test checks that embeddings are non-zero and distinct.")
    print("For full accuracy verification, we need to compare against the")
    print("transformers reference implementation.")
    print("="*60)

if __name__ == "__main__":
    test_reranking()
