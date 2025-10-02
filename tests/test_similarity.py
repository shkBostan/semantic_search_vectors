"""
File: test_similarity.py
Created on Oct, 2025
Author: s Bostan

Description:
    Test script to demonstrate generating embeddings and computing
    semantic similarity between sentences using the modules in src/.
    Licensed under the Apache License 2.0.
"""
from src.embeddings import get_embedding
from src.similarity import cosine_similarity

def run_tests():
    sentences = [
        "I love machine learning.",
        "I enjoy deep learning.",
        "The weather is sunny today."
    ]
    embeddings = [get_embedding(s) for s in sentences]
    for i, s in enumerate(sentences[1:], start=1):
        sim = cosine_similarity(embeddings[0], embeddings[i])
        print(f"Similarity between '{sentences[0]}' and '{s}': {sim:.2f}")
