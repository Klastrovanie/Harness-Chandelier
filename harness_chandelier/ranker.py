import cudf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List

from .topic import extract_topics
from .weights import calculate_weighted_score, scale_wgt, compute_delta_time, normalize_wgt
from .graph import run_pagerank


def generate_realistic_timestamps(n_messages: int, base_time: datetime = None) -> List[datetime]:
    if base_time is None:
        base_time = datetime.now()
    timestamps = [base_time]
    for _ in range(n_messages - 1):
        r = np.random.random()
        if r < 0.5:
            gap = np.random.randint(5, 60)
        elif r < 0.8:
            gap = np.random.randint(60, 300)
        else:
            gap = np.random.randint(600, 3600)
        timestamps.append(timestamps[-1] + timedelta(seconds=int(gap)))
    return timestamps


@dataclass
class RankerResult:
    topic_labels: list
    main_topic: int
    pagerank: pd.DataFrame
    edge_df: pd.DataFrame


class HarnessChandelier:
    """
    Weight Guidelines:
        delta_time: +0.2  -> Topics user keeps returning to (true intent)
        delta_time: -0.2  -> Frequently switching topics (noise/blockers)
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        scale_factor: int = 1000,
        scaler: str = "scale_and_round",
        n_neighbors: int = 3,
        min_cluster_size: int = 3, # if you increase this, you will have less topics.
        base_time: Optional[datetime] = None,
        random_seed: int = 42
    ):
        self.weights = weights or {"delta_time": +0.2, "transition_count": 1.0}
        self.scale_factor = scale_factor
        self.scaler = scaler
        self.n_neighbors = n_neighbors
        self.min_cluster_size = min_cluster_size
        self.base_time = base_time or datetime.now()
        self.random_seed = random_seed

    def fit(self, messages: list, timestamps: Optional[List[datetime]] = None) -> RankerResult:
        np.random.seed(self.random_seed)

        topics, probs, _ = extract_topics(
            messages,
            n_neighbors=self.n_neighbors,
            min_cluster_size=self.min_cluster_size
        )

        if timestamps is None:
            timestamps = generate_realistic_timestamps(
                n_messages=len(messages),
                base_time=self.base_time
            )

        if len(timestamps) != len(messages):
            raise ValueError(f"timestamps length ({len(timestamps)}) must match messages length ({len(messages)})")

        edges = []
        for i in range(len(topics) - 1):
            edges.append({
                'src': topics[i],
                'dst': topics[i + 1],
                'timestamp': timestamps[i]
            })

        pdf = pd.DataFrame(edges)
        pdf['timestamp'] = pd.to_datetime(pdf['timestamp'])
        pdf = compute_delta_time(pdf)
        pdf['transition_count'] = pdf.groupby(['src', 'dst'])['src'].transform('count')
        pdf = calculate_weighted_score(pdf, self.weights)
        pdf = normalize_wgt(pdf)
        pdf = scale_wgt(pdf, self.scaler, self.scale_factor)

        gdf = cudf.from_pandas(pdf[['src', 'dst', 'wgt']].astype('int32'))
        result = run_pagerank(gdf)

        return RankerResult(
            topic_labels=topics,
            main_topic=result.main_topic,
            pagerank=result.pagerank,
            edge_df=pdf
        )