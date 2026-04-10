import cudf
import cugraph
import pandas as pd
from dataclasses import dataclass


@dataclass
class GraphResult:
    pagerank: pd.DataFrame
    main_topic: int


def build_graph(gdf: cudf.DataFrame) -> cugraph.Graph:
    G = cugraph.Graph(directed=True) 
    G.from_cudf_edgelist(
        gdf,
        source='src',
        destination='dst',
        edge_attr='wgt',
        store_transposed=True
    )
    return G


def run_pagerank(gdf: cudf.DataFrame) -> GraphResult:
    G = build_graph(gdf)
    
    pagerank_scores = cugraph.pagerank(G)
    pagerank_pd = pagerank_scores.to_pandas().sort_values('pagerank', ascending=False).reset_index(drop=True)
    main_topic = int(pagerank_pd.iloc[0]['vertex'])
    return GraphResult(pagerank=pagerank_pd, main_topic=main_topic)