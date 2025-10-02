"""
File: similarity.py
Created on Oct, 2025
Author: s Bostan

Description:
    This module provides functions to compute similarity between embeddings
    using cosine similarity from sentence-transformers.
    Licensed under the Apache License 2.0.
"""

from sentence_transformers import util


def cosine_similarity(emb1, emb2) -> float:
    """
    Compute cosine similarity between two embeddings.

    :param emb1: first embedding
    :param emb2: second embedding
    :return: similarity score as float
    """
    return util.cos_sim(emb1, emb2).item()
