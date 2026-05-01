"""
lang/en.py
English language configuration for Harness-Chandelier.
BERTopic handles English stopwords internally, so no custom
vectorizer is needed. This file exists for structural consistency.
"""
from sklearn.feature_extraction.text import CountVectorizer


def get_vectorizer() -> CountVectorizer:
    """
    Returns a CountVectorizer with English stopwords.
    BERTopic's default English stopword handling is used.
    """
    return CountVectorizer(stop_words="english")
