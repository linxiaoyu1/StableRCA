import time
from typing import Any, Callable

import pandas as pd
import networkx as nx

from algorithms.stable_rca_model.MDS_detection import MDS_detector
from algorithms.stable_rca_model.utils import (
    get_all_descendants,
    get_markov_blanket,
)
from algorithms.stable_rca_model.CDS_detection import (
    stable_variable_selection,
    regression_cds_test,
    classification_cds_test,
)


def apply_stable_rca(
    args: Any,
    graph: nx.DiGraph,
    root_cause: str,
    normal_data: pd.DataFrame,
    anomaly_data: pd.DataFrame,
):
    """
    Apply StableRCA to detect root-cause candidates.

    Pipeline:
        Phase 1: Marginal distribution shift detection.
        Phase 2: Markov boundary / stable variable selection.
        Phase 3: Conditional distribution shift detection.

    Returns
    -------
    scores : dict
        RCA score for each variable. Non-shifted variables are assigned -100.0.

    runtime_info : dict
        Runtime statistics for each phase and each shifted node.
    """
    total_start = time.perf_counter()
    verbose = getattr(args, "verbose", False)

    scores = {}
    runtime_info = {
        "total_time": None,
        "phase_1_mds_time": 0.0,
        "phase_2_mb_total_time": 0.0,
        "phase_3_cds_total_time": 0.0,
        "continuous_total_time": 0.0,
        "discrete_total_time": 0.0,
        "per_node": {},
        "n_shift_nodes": 0,
        "n_continuous_shift_nodes": 0,
        "n_discrete_shift_nodes": 0,
    }

    # =========================
    # Phase 1: Marginal distribution shift detection
    # =========================
    phase1_start = time.perf_counter()

    mds_detector = MDS_detector(args)
    shift_col_list = mds_detector.test(normal_data, anomaly_data)

    runtime_info["phase_1_mds_time"] = time.perf_counter() - phase1_start

    continuous_cols = [
        col for col in shift_col_list if col in mds_detector.continuous_cols
    ]
    discrete_cols = [
        col for col in shift_col_list if col in mds_detector.discrete_cols
    ]

    runtime_info["n_shift_nodes"] = len(shift_col_list)
    runtime_info["n_continuous_shift_nodes"] = len(continuous_cols)
    runtime_info["n_discrete_shift_nodes"] = len(discrete_cols)

    # Assign a low default score to variables without marginal shift.
    shift_cols = set(shift_col_list)
    for col in normal_data.columns:
        if col not in shift_cols:
            scores[col] = -100.0

    if verbose:
        gt_abnormal_nodes = get_all_descendants(graph, root_cause) | {root_cause}

        print(f"Root cause: {root_cause}")
        print(f"Detected abnormal nodes: {shift_col_list}")
        print(f"Ground-truth abnormal nodes: {gt_abnormal_nodes}")
        print(f"Continuous abnormal nodes: {continuous_cols}")
        print(f"Discrete abnormal nodes: {discrete_cols}")
        print(f"Phase 1 MDS time: {runtime_info['phase_1_mds_time']:.6f}s")

    def process_shifted_node(
        node: str,
        node_type: str,
        task: str,
        prediction_model: str,
        cds_test_fn: Callable,
        cds_kwargs: dict | None = None,
    ) -> None:
        """
        Run Phase 2 and Phase 3 for one shifted node.
        """
        cds_kwargs = cds_kwargs or {}
        node_start = time.perf_counter()

        # -------------------------
        # Phase 2: Stable variable selection
        # -------------------------
        phase2_start = time.perf_counter()

        selected_feature_names, feature_weights = stable_variable_selection(
            normal_data,
            args,
            node,
            task=task,
            prediction_model=prediction_model,
        )

        phase2_time = time.perf_counter() - phase2_start
        runtime_info["phase_2_mb_total_time"] += phase2_time

        if verbose:
            gt_mb_nodes = get_markov_blanket(graph, node)

            print(f"\nTarget: {node}")
            print(f"Variable type: {node_type}")
            print(f"GT Markov blanket: {gt_mb_nodes}")
            print(f"Selected features: {selected_feature_names}")
            print(f"Feature weights: {feature_weights}")
            print(f"Phase 2 time for {node}: {phase2_time:.6f}s")

        # -------------------------
        # Phase 3: Conditional distribution shift detection
        # -------------------------
        phase3_start = time.perf_counter()

        rca_score = cds_test_fn(
            normal_data,
            anomaly_data,
            node,
            selected_feature_names,
            **cds_kwargs,
        )

        phase3_time = time.perf_counter() - phase3_start
        runtime_info["phase_3_cds_total_time"] += phase3_time

        scores[node] = rca_score

        node_total_time = time.perf_counter() - node_start
        runtime_info["per_node"][node] = {
            "type": node_type,
            "phase_2_mb_time": phase2_time,
            "phase_3_cds_time": phase3_time,
            "total_node_time": node_total_time,
            "n_selected_features": len(selected_feature_names),
        }

        if verbose:
            print(f"Node {node}, RCA score: {rca_score}")
            print(f"Phase 3 time for {node}: {phase3_time:.6f}s")
            print(f"Total time for {node}: {node_total_time:.6f}s")

    # =========================
    # Continuous variables
    # =========================
    continuous_start = time.perf_counter()

    for node in continuous_cols:
        process_shifted_node(
            node=node,
            node_type="continuous",
            task="regression",
            prediction_model=args.stable_prediction_model_continuous,
            cds_test_fn=regression_cds_test,
        )

    runtime_info["continuous_total_time"] = time.perf_counter() - continuous_start

    # =========================
    # Discrete variables
    # =========================
    discrete_start = time.perf_counter()

    for node in discrete_cols:
        process_shifted_node(
            node=node,
            node_type="discrete",
            task="classification",
            prediction_model=args.stable_prediction_model_categorical,
            cds_test_fn=classification_cds_test,
            cds_kwargs={
                "default_large_score": args.default_large_score,
            },
        )

    runtime_info["discrete_total_time"] = time.perf_counter() - discrete_start
    runtime_info["total_time"] = time.perf_counter() - total_start

    if verbose:
        print("\n===== Runtime Summary =====")
        print(f"Phase 1 MDS time: {runtime_info['phase_1_mds_time']:.6f}s")
        print(f"Phase 2 MB total time: {runtime_info['phase_2_mb_total_time']:.6f}s")
        print(f"Phase 3 CDS total time: {runtime_info['phase_3_cds_total_time']:.6f}s")
        print(f"Continuous total time: {runtime_info['continuous_total_time']:.6f}s")
        print(f"Discrete total time: {runtime_info['discrete_total_time']:.6f}s")
        print(f"Total StableRCA time: {runtime_info['total_time']:.6f}s")

    return scores, runtime_info