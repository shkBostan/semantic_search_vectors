"""
File: embeddings.py
Created on Oct, 2025
Author: s Bostan

Description:
    This module provides functions to convert sentences into vector embeddings
    using pre-trained SentenceTransformer models.
    Licensed under the Apache License 2.0.
"""
from sentence_transformers import SentenceTransformer

# Load the model once and reuse
_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(sentence: str):
    """
    Convert a sentence into a vector embedding.
    :param sentence: input sentence
    :return: tensor embedding
    """
    return _model.encode(sentence, convert_to_tensor=True)
