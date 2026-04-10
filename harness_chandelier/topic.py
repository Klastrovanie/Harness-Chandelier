from bertopic import BERTopic
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN


def build_topic_model(n_neighbors: int = 3, n_components: int = 2, min_cluster_size: int = 2):
    """
    Build BERTopic model using cuML UMAP and HDBSCAN (GPU-accelerated).
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

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="english",
        calculate_probabilities=True,
        verbose=False
    )

    return topic_model


def extract_topics(messages: list, n_neighbors: int = 3, min_cluster_size: int = 2):
    """
    Extract topics from messages using BERTopic + cuML.

    Returns:
        topics: list of topic labels per message
        probs: probability matrix
        topic_model: fitted BERTopic model
    """
    topic_model = build_topic_model(
        n_neighbors=n_neighbors,
        min_cluster_size=min_cluster_size
    )
    topics, probs = topic_model.fit_transform(messages)
    return topics, probs, topic_model
