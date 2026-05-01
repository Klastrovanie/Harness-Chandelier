from bertopic import BERTopic
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from sentence_transformers import SentenceTransformer

from .lang import get_vectorizer

_cached_model = None
_cached_embedding_model = None


def get_embedding_model():
    global _cached_embedding_model
    if _cached_embedding_model is None:
        _cached_embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
    return _cached_embedding_model


def build_topic_model(
    n_neighbors: int = 3,
    n_components: int = 2,
    min_cluster_size: int = 2,
    lang: str = "en"
):
    """
    Build BERTopic model using cuML UMAP and HDBSCAN (GPU-accelerated).

    Args:
        n_neighbors: UMAP n_neighbors. Recommended: 3 for English, 5 for Korean.
        n_components: UMAP output dimensions.
        min_cluster_size: HDBSCAN min cluster size.
        lang: Language code. "en" or "ko". Affects stopword filtering only.
              Even with lang="ko", English text is fully supported.
    """
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        prediction_data=True
    )

    vectorizer = get_vectorizer(lang)

    topic_model = BERTopic(
        embedding_model=get_embedding_model(),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        language="multilingual",
        calculate_probabilities=True,
        verbose=False
    )

    return topic_model


def extract_topics(
    messages: list,
    n_neighbors: int = 3,
    min_cluster_size: int = 2,
    lang: str = "en"
):
    """
    Extract topics from messages using BERTopic + cuML.

    Args:
        messages: List of messages to extract topics from.
        n_neighbors: UMAP n_neighbors.
        min_cluster_size: HDBSCAN min cluster size.
        lang: Language code. "en" or "ko".

    Returns:
        topics: list of topic labels per message
        probs: probability matrix
        topic_model: fitted BERTopic model
    """
    global _cached_model

    if _cached_model is None:
        _cached_model = build_topic_model(
            n_neighbors=n_neighbors,
            min_cluster_size=min_cluster_size,
            lang=lang
        )

    # multilingual-e5-base requires "query: " prefix for best performance
    prefixed = [f"query: {m}" for m in messages]

    topics, probs = _cached_model.fit_transform(prefixed)
    return topics, probs, _cached_model
