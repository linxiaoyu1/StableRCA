from __future__ import annotations

import random
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from torch.distributions import Normal

from data_scm.scm_data_simulation import SCM
from data_scm.utils import get_all_descendants
from data_simulators.xges_discovery import df_to_xges_digraph


# =============================================================================
# Candidate and target node selection
# =============================================================================

def _get_intervention_candidates(
    dag: nx.DiGraph,
    mode: str,
) -> list:
    """
    Return candidate intervention nodes according to graph position.

    mode:
        - "non_root": nodes with at least one parent.
        - "root": nodes with no parents.
        - "any": all nodes.
    """
    if mode == "non_root":
        return [node for node in dag.nodes if dag.in_degree(node) != 0]

    if mode == "root":
        return [node for node in dag.nodes if dag.in_degree(node) == 0]

    if mode == "any":
        return list(dag.nodes)

    raise ValueError(
        f"Unknown intervention_node_mode={mode}. "
        "Expected one of ['non_root', 'root', 'any']."
    )

def _filter_by_descendant_count(
    dag: nx.DiGraph,
    candidate_nodes: list,
    min_descendants: int = 0,
    max_descendants: Optional[int] = None,
) -> tuple[list, dict]:
    """
    Keep candidate intervention nodes whose descendant counts satisfy the
    specified range.
    """
    filtered_nodes = []
    descendant_count = {}

    for node in candidate_nodes:
        num_descendants = len(get_all_descendants(dag, node))
        descendant_count[node] = num_descendants

        if num_descendants < min_descendants:
            continue

        if max_descendants is not None and num_descendants > max_descendants:
            continue

        filtered_nodes.append(node)

    return filtered_nodes, descendant_count


def _select_target_node(
    dag: nx.DiGraph,
    candidate_nodes: set | list,
    mode: str,
):
    """
    Select a target node from candidate abnormal nodes.

    mode:
        - "leaf": select from leaf nodes.
        - "non_leaf": select from non-leaf nodes.
        - "random": select from all candidates.
    """
    candidate_nodes = list(candidate_nodes)

    if len(candidate_nodes) == 0:
        return None

    if mode == "leaf":
        pool = [node for node in candidate_nodes if dag.out_degree(node) == 0]
    elif mode == "non_leaf":
        pool = [node for node in candidate_nodes if dag.out_degree(node) > 0]
    elif mode == "random":
        pool = candidate_nodes
    else:
        raise ValueError(
            f"Unknown target_node_mode={mode}. "
            "Expected one of ['leaf', 'non_leaf', 'random']."
        )

    if len(pool) == 0:
        return None

    return random.choice(pool)


# =============================================================================
# Graph corruption and benchmark graph construction
# =============================================================================

def _try_add_acyclic_edge(
    graph: nx.DiGraph,
    source,
    target,
) -> bool:
    """
    Try to add edge source -> target while preserving DAG acyclicity.

    Return True if the edge is added successfully.
    """
    if source == target or graph.has_edge(source, target):
        return False

    graph.add_edge(source, target)

    if nx.is_directed_acyclic_graph(graph):
        return True

    graph.remove_edge(source, target)
    return False


def corrupt_dag(
    dag: nx.DiGraph,
    delete_frac: float = 0.15,
    reverse_frac: float = 0.10,
    add_frac: float = 0.05,
    seed: Optional[int] = None,
) -> nx.DiGraph:
    """
    Corrupt a DAG by randomly deleting, reversing, and adding edges.

    The returned graph remains acyclic. Data are still generated from the
    original ground-truth SCM.
    """
    rng = random.Random(seed)
    graph = dag.copy()

    initial_edges = list(graph.edges())
    initial_num_edges = len(initial_edges)

    num_delete = int(round(delete_frac * initial_num_edges))
    num_reverse = int(round(reverse_frac * initial_num_edges))
    num_add = int(round(add_frac * initial_num_edges))

    # 1. Random edge deletion
    edges = list(graph.edges())
    rng.shuffle(edges)

    deleted_edges = edges[: min(num_delete, len(edges))]
    graph.remove_edges_from(deleted_edges)

    # 2. Random edge reversal while preserving acyclicity
    edges = list(graph.edges())
    rng.shuffle(edges)

    reversed_edges = 0

    for source, target in edges:
        if reversed_edges >= num_reverse:
            break

        if not graph.has_edge(source, target):
            continue

        graph.remove_edge(source, target)

        if _try_add_acyclic_edge(graph, target, source):
            reversed_edges += 1
        else:
            graph.add_edge(source, target)

    # 3. Random false-positive edge addition while preserving acyclicity
    nodes = list(graph.nodes())

    candidate_edges = [
        (source, target)
        for source in nodes
        for target in nodes
        if source != target and not graph.has_edge(source, target)
    ]

    rng.shuffle(candidate_edges)

    added_edges = 0

    for source, target in candidate_edges:
        if added_edges >= num_add:
            break

        if _try_add_acyclic_edge(graph, source, target):
            added_edges += 1

    return graph


def _get_graph_for_benchmark(
    args: Any,
    true_dag: nx.DiGraph,
    df_obs_data: pd.DataFrame,
) -> nx.DiGraph:
    """
    Return the graph used by RCA methods.

    Supported graph modes:
        - "true": use the ground-truth DAG.
        - "xges": use the XGES-estimated DAG.
        - "corrupted": use a manually corrupted version of the ground-truth DAG.
    """
    graph_mode = getattr(args, "graph_mode", "true")

    if graph_mode == "true":
        return true_dag.copy()

    if graph_mode == "xges":
        return df_to_xges_digraph(df_obs_data)

    if graph_mode == "corrupted":
        return corrupt_dag(
            true_dag,
            delete_frac=float(getattr(args, "corrupt_delete_frac", 0.15)),
            reverse_frac=float(getattr(args, "corrupt_reverse_frac", 0.10)),
            add_frac=float(getattr(args, "corrupt_add_frac", 0.05)),
            seed=getattr(args, "corrupt_graph_seed", None),
        )

    raise ValueError(
        f"Unknown graph_mode={graph_mode}. "
        "Expected one of ['true', 'xges', 'corrupted']."
    )


# =============================================================================
# Main synthetic data generator
# =============================================================================

def synthetic_data_generator(args: Any) -> dict:
    """
    Generate one synthetic RCA benchmark instance.

    Noise-type auto-sampling is handled inside SCM.

    Returns
    -------
    dict
        Dictionary containing:
        - graph
        - training_sample
        - anomaly_sample
        - root_cause
        - target_node
    """
    intervention_node_mode = getattr(args, "intervention_node_mode", "non_root")
    target_node_mode = getattr(args, "target_node_mode", "leaf")

    min_descendants = int(getattr(args, "min_intervention_descendants", 0))
    max_descendants = getattr(args, "max_intervention_descendants", None)

    if max_descendants is not None:
        max_descendants = int(max_descendants)
        if max_descendants < 0:
            max_descendants = None

    max_generation_attempts = int(getattr(args, "max_generation_attempts", 1000))
    last_candidate_summary = None

    for _ in range(max_generation_attempts):
        scm = SCM(args, device="cpu")

        raw_candidate_nodes = _get_intervention_candidates(
            dag=scm.dag,
            mode=intervention_node_mode,
        )

        candidate_nodes, _ = _filter_by_descendant_count(
            dag=scm.dag,
            candidate_nodes=raw_candidate_nodes,
            min_descendants=min_descendants,
            max_descendants=max_descendants,
        )

        last_candidate_summary = {
            "raw_candidates": len(raw_candidate_nodes),
            "filtered_candidates": len(candidate_nodes),
            "min_descendants": min_descendants,
            "max_descendants": max_descendants,
        }

        if len(candidate_nodes) < args.n_intervention_nodes:
            continue

        # Generate observational data.
        obs_data = scm(args.n_sample_normal)
        df_obs_data = pd.DataFrame(obs_data, columns=list(scm.dag.nodes))

        # Select intervention nodes.
        intervention_nodes = list(
            np.random.choice(
                candidate_nodes,
                size=args.n_intervention_nodes,
                replace=False,
            )
        )

        # Build intervention dict
        intervention_dict = {}
        for node in intervention_nodes:
            if args.intervention_type == "hard":
                intervention_dict[node] = (
                    df_obs_data[node].mean()
                    + args.hard_intervention_magnitude * df_obs_data[node].std()
                )
            elif args.intervention_type == "soft_function":
                intervention_dict[node] = {
                    "function_type": scm.function_type,
                    "function_params": scm.function_params,
                }
            elif args.intervention_type == "soft_noise":
                intervention_dict[node] = {
                    "noise_type": np.random.choice(["Laplace", "Gumbel"], size=1)[0],
                    "noise_std": np.abs(np.random.randn()),
                }
            elif args.intervention_type == "soft_distribution":
                obs_mean = float(df_obs_data[node].mean())
                obs_std = float(df_obs_data[node].std())

                # Strength of the distributional anomaly:
                # loc = mean + magnitude * std
                magnitude = float(getattr(
                    args,
                    "soft_distribution_magnitude",
                    getattr(args, "hard_intervention_magnitude", 1.5)
                ))

                # Variance shrinkage. For example, 0.3 means the interventional
                # std is 30% of the observational std.
                std_scale = float(getattr(args, "soft_distribution_std_scale", 0.3))
                min_std = float(getattr(args, "soft_distribution_min_std", 1e-6))

                # Optional direction control.
                # positive: mean + magnitude * std
                # negative: mean - magnitude * std
                # random: randomly choose the direction per intervention node
                shift_direction = getattr(args, "soft_distribution_shift_direction", "positive")

                if shift_direction == "positive":
                    direction = 1.0
                elif shift_direction == "negative":
                    direction = -1.0
                elif shift_direction == "random":
                    direction = float(np.random.choice([-1.0, 1.0]))
                else:
                    raise ValueError(
                        f"Unknown soft_distribution_shift_direction={shift_direction}. "
                        "Expected one of ['positive', 'negative', 'random']."
                    )

                int_mean = obs_mean + direction * magnitude * obs_std
                int_std = max(std_scale * obs_std, min_std)

                intervention_dict[node] = Normal(
                    float(int_mean),
                    float(int_std),
                )
            else:
                raise ValueError(f"Unknown intervention type: {args.ours_intervention_type}")

        # Generate interventional data.
        int_data = scm.sample_intervention(
            args.n_sample_abnormal,
            args.intervention_type,
            intervention_dict,
        )

        df_int_data = pd.DataFrame(int_data, columns=list(scm.dag.nodes))

        # Select target node from affected nodes.
        root_causes = intervention_nodes

        intervention_descendants = {
            node: get_all_descendants(scm.dag, node) | {node}
            for node in root_causes
        }

        gt_abnormal_nodes = set().union(*intervention_descendants.values())
        affected_descendants_only = gt_abnormal_nodes - set(root_causes)

        target_node = _select_target_node(
            dag=scm.dag,
            candidate_nodes=affected_descendants_only,
            mode=target_node_mode,
        )

        if target_node is None:
            target_node = _select_target_node(
                dag=scm.dag,
                candidate_nodes=gt_abnormal_nodes - set(root_causes),
                mode="random",
            )

        if target_node is None:
            target_node = _select_target_node(
                dag=scm.dag,
                candidate_nodes=gt_abnormal_nodes,
                mode="random",
            )

        if target_node is None:
            continue

        # Build benchmark graph and rename variables.
        rename_map = {node: f"X{node}" for node in scm.dag.nodes}

        raw_graph = _get_graph_for_benchmark(
            args=args,
            true_dag=scm.dag,
            df_obs_data=df_obs_data,
        )

        graph = nx.relabel_nodes(raw_graph, rename_map)

        training_sample = df_obs_data.rename(columns=rename_map)
        anomaly_sample = df_int_data.rename(columns=rename_map)

        root_cause = (
            rename_map[root_causes[0]]
            if len(root_causes) == 1
            else [rename_map[node] for node in root_causes]
        )

        return {
            "graph": graph,
            "training_sample": training_sample,
            "anomaly_sample": anomaly_sample,
            "root_cause": root_cause,
            "target_node": rename_map[target_node],
        }

    raise RuntimeError(
        "Failed to generate a valid synthetic RCA instance after "
        f"{max_generation_attempts} attempts. "
        f"Last candidate summary: {last_candidate_summary}. "
        "Try relaxing --min_intervention_descendants / "
        "--max_intervention_descendants, increasing --n_edges, "
        "or increasing --n_nodes."
    )