# ./algorithms/rcg_0.py
"""RCG-0 wrapper adapted from /rcg implementation.

This keeps the essential RCG ranking logic and constructs the prior graph
with k=0 using k-PC over d-separation on the provided causal DAG.
"""

import copy
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import networkx as nx

from causallearn.graph.Edge import Edge
from causallearn.graph.Graph import Graph
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from sklearn.preprocessing import KBinsDiscretizer

F_NODE = "F-node"


def _rcg_root() -> str:
    return str(Path(__file__).resolve().parents[1] / "rcg")


def _ensure_rcg_path() -> None:
    rcg_root = _rcg_root()
    if rcg_root not in sys.path:
        sys.path.insert(0, rcg_root)


def _learn_k0_graph_from_dag(graph: nx.DiGraph) -> GeneralGraph:
    """Learn k=0 prior graph using the same k-PC implementation as /rcg."""
    _ensure_rcg_path()
    from para_kpc.kPC_fas import kpc

    node_names = list(graph.nodes)
    g, _ = kpc(
        np.array([[]]),
        independence_test_method="d_separation",
        true_dag=graph,
        k=0,
        n=len(node_names),
        node_names=node_names,
        parallel=True,
        s=None,
        batch=None,
        p_cores=1,
    )
    return g


def _compute_joint_probabilities(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    total_count = len(df)
    if total_count == 0:
        out = df[columns].copy()
        out["counts"] = []
        out["probability"] = []
        return out

    joint_prob_df = df.groupby(columns).size().reset_index(name="counts")
    joint_prob_df["probability"] = joint_prob_df["counts"] / total_count
    return joint_prob_df


def _mutual_information(df: pd.DataFrame, x: str, y: str) -> float:
    joint = _compute_joint_probabilities(df, [x, y])
    x_prob = _compute_joint_probabilities(df, [x])
    y_prob = _compute_joint_probabilities(df, [y])

    merged = pd.merge(joint, x_prob, on=[x], suffixes=("", "_x"))
    merged = pd.merge(merged, y_prob, on=[y], suffixes=("", "_y"))
    merged["mi_contrib"] = merged["probability"] * np.log2(
        merged["probability"] / (merged["probability_x"] * merged["probability_y"])
    )
    return float(merged["mi_contrib"].sum())


def _conditional_mutual_information(df: pd.DataFrame, x: str, y: str, z: List[str]) -> float:
    if not z:
        return _mutual_information(df, x, y)

    joint = _compute_joint_probabilities(df, [x, y] + z)
    xz = _compute_joint_probabilities(df, [x] + z)
    yz = _compute_joint_probabilities(df, [y] + z)
    z_prob = _compute_joint_probabilities(df, z)

    merged = pd.merge(joint, xz, on=[x] + z, suffixes=("", "_xz"))
    merged = pd.merge(merged, yz, on=[y] + z, suffixes=("", "_yz"))
    merged = pd.merge(merged, z_prob, on=z, suffixes=("", "_z"))

    merged["cmi_contrib"] = merged["probability"] * np.log2(
        merged["probability"] * merged["probability_z"]
        / (merged["probability_xz"] * merged["probability_yz"])
    )
    return float(merged["cmi_contrib"].sum())


def _add_fnode(normal_df: pd.DataFrame, anomalous_df: pd.DataFrame) -> pd.DataFrame:
    normal = normal_df.copy()
    anomalous = anomalous_df.copy()
    normal[F_NODE] = 0
    anomalous[F_NODE] = 1
    return pd.concat([normal, anomalous], ignore_index=True)


def _discretize_shared(normal_df: pd.DataFrame, anomalous_df: pd.DataFrame, bins: int = 5):
    """Discretize continuous features using shared bins across normal/anomalous data."""
    combined = pd.concat([normal_df, anomalous_df], ignore_index=True)
    discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')

    # Keep behavior stable when features are low-variance/constant.
    with np.errstate(all='ignore'):
        disc = discretizer.fit_transform(combined)

    disc_df = pd.DataFrame(disc, columns=combined.columns).astype(int)
    n = normal_df.shape[0]
    return disc_df.iloc[:n].reset_index(drop=True), disc_df.iloc[n:].reset_index(drop=True)


class _Scorer:
    def __init__(self, df: pd.DataFrame):
        self.df = df  # Assumes F-node is already added
        self._cache = {}

    def get_score(self, y: str, z: Set[str] | None = None) -> float:
        z = set() if z is None else z
        xz_key = (y, frozenset(z))
        if xz_key in self._cache:
            return self._cache[xz_key]

        if len(z) == 0:
            score = _mutual_information(self.df, F_NODE, y)
        else:
            score = _conditional_mutual_information(self.df, F_NODE, y, list(z))

        self._cache[xz_key] = score
        return score

    def compute_mi(self):
        return [(y, self.get_score(y)) for y in self.df.columns if y != F_NODE]

    def compute_cmi_on_possible_parents(self, graph: GeneralGraph):
        scores = []
        for y in [c for c in self.df.columns if c != F_NODE]:
            poss_pa = [x.name for x in _find_possible_parents(graph, graph.get_node(y))]
            scores.append((y, self.get_score(y, z=set(poss_pa))))
        return scores


def _update_edges(edges: List[tuple], graph: GeneralGraph) -> None:
    for old_edge, new_edge in edges:
        graph.remove_edge(old_edge)
        if new_edge is not None:
            graph.add_edge(new_edge)


def _is_possible_parent(graph: Graph, potential_parent_node, child_node) -> bool:
    if graph.node_map[potential_parent_node] == graph.node_map[child_node]:
        return False
    if not graph.is_adjacent_to(potential_parent_node, child_node):
        return False
    if graph.get_endpoint(child_node, potential_parent_node) == Endpoint.ARROW:
        return False
    return True


def _find_possible_parents(graph: Graph, child_node, en_nodes=None):
    if en_nodes is None:
        nodes = graph.get_nodes()
        en_nodes = [node for node in nodes if graph.node_map[node] != graph.node_map[child_node]]
    return [parent_node for parent_node in en_nodes if _is_possible_parent(graph, parent_node, child_node)]


def _local_run(df: pd.DataFrame, graph: GeneralGraph, l: int):
    scorer = _Scorer(df)
    best_ranking = []
    alphas = sorted(set([x[1] for x in scorer.compute_mi()]))

    for alpha in alphas:
        g = copy.deepcopy(graph)

        new_edges = []
        for x in g.get_nodes():
            for y in g.get_adjacent_nodes(x):
                fx = scorer.get_score(x.name)
                fy = scorer.get_score(y.name)
                if fx < alpha <= fy:
                    old_edge = g.get_edge(x, y)
                    new_edge = None
                    if g.is_undirected_from_to(x, y):
                        new_edge = Edge(x, y, Endpoint.TAIL, Endpoint.ARROW)
                    new_edges.append((old_edge, new_edge))
        _update_edges(new_edges, g)

        cmi = scorer.compute_cmi_on_possible_parents(g)
        sorted_cmi = sorted(cmi, key=lambda t: t[1], reverse=True)

        # Consistency check
        for node, _score in sorted_cmi[:l]:
            if scorer.get_score(node) < alpha:
                return best_ranking

        best_ranking = sorted_cmi

    return best_ranking


def apply_rcg_0(graph: nx.DiGraph, normal_data: pd.DataFrame, anomaly_data: pd.DataFrame, l: int | None = None) -> Dict[str, float]:
    """Apply RCG-0 and return metric -> score."""
    common_cols = [c for c in normal_data.columns if c in anomaly_data.columns]
    if not common_cols:
        return {}

    normal_df = normal_data[common_cols].copy()
    anomal_df = anomaly_data[common_cols].copy()

    # Keep only nodes present in both data and graph.
    graph_nodes = [n for n in graph.nodes if n in normal_df.columns]
    if not graph_nodes:
        return {}

    normal_df = normal_df[graph_nodes]
    anomal_df = anomal_df[graph_nodes]

    # RCG is designed for discrete data; discretize continuous samples first.
    normal_df, anomal_df = _discretize_shared(normal_df, anomal_df, bins=5)

    subgraph = graph.subgraph(graph_nodes).copy()
    prior_graph = _learn_k0_graph_from_dag(subgraph)

    df = _add_fnode(normal_df, anomal_df)
    top_l = len(graph_nodes) if l is None else l
    ranked = _local_run(df, prior_graph, top_l)
    return {node: float(score) for node, score in ranked}
