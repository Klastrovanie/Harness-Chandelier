"""
lang/__init__.py
Language configuration package for Harness-Chandelier.

Supported languages:
    "en" - English (default)
    "ko" - Korean (English mixed text also supported)
"""
from sklearn.feature_extraction.text import CountVectorizer


# default parameters for each language
# n_neighbors: higher values consider wider context (5 recommended for Korean as sentences are shorter)
LANG_DEFAULTS = {
    "en": {"n_neighbors": 3, "min_cluster_size": 3},
    "ko": {"n_neighbors": 5, "min_cluster_size": 3},
}


def get_lang_defaults(lang: str = "en") -> dict:
    """
    Returns recommended default parameters for the given language.

    Args:
        lang: Language code. "en" or "ko".

    Returns:
        dict with n_neighbors and min_cluster_size.
    """
    return LANG_DEFAULTS.get(lang, LANG_DEFAULTS["en"])


def get_vectorizer(lang: str = "en") -> CountVectorizer:
    """
    Returns a CountVectorizer for the given language.

    Args:
        lang: Language code. "en" or "ko".

    Returns:
        CountVectorizer with appropriate stopwords.
    """
    if lang == "ko":
        from .ko import get_vectorizer as _get
    else:
        from .en import get_vectorizer as _get

    return _get()
