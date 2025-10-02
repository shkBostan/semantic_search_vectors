"""
Created on Oct, 2025
Author: s Bostan

Description:
    This is the main entry point for the semantic_search_vectors project.
    It executes the test suite defined in tests/test_similarity.py
    to verify that embeddings and similarity calculations work correctly.

    Licensed under the Apache License 2.0.
"""
from tests.test_similarity import run_tests

if __name__ == "__main__":
    print("Running semantic search tests...")
    run_tests()
