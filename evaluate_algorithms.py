# ./evaluate_algorithms.py
from __future__ import annotations

import math
import random
import time
from typing import Any, Callable, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from dowhy.gcm import RescaledMedianCDFQuantileScorer
from tqdm import tqdm

from algorithms import (
    apply_baro,
    apply_cholesky,
    apply_circa,
    apply_counterfactual_contribution,
    apply_epsilon_diagnosis,
    apply_rcd,
    apply_rcg_0,
    apply_score_ordering,
    apply_simple_traversal,
    apply_smooth_traversal,
    apply_stable_rca,
)


# =============================================================================
# Metric helpers
# =============================================================================

def _as_root_cause_list(root_cause: str | Sequence[str]) -> list[str]:
    """Normalize a single root cause or multiple root causes into a list."""
    if isinstance(root_cause, str):
        return [root_cause]
    return list(root_cause)


def top_k_metrics(
    scores: dict[str, float],
    root_causes: Sequence[str],
    k: int,
    adjust_for_ties: bool = False,
) -> dict[str, float]:
    """
    Compute Top-k precision, recall, F1, and NDCG.

    Parameters
    ----------
    scores:
        Mapping from variable name to RCA score. Higher is better.

    root_causes:
        Ground-truth root-cause variables.

    k:
        Evaluation cutoff.

    adjust_for_ties:
        If True, include all variables tied with the k-th score.
        If False, randomly sample among variables tied at the k-th score.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    if len(root_causes) == 0:
        raise ValueError("root_causes must contain at least one variable.")

    if len(scores) == 0:
        return _zero_ranking_metrics()

    root_cause_set = set(root_causes)

    # If the algorithm does not score at least one root cause, treat the run as
    # a failure for root-cause recovery.
    if not root_cause_set.issubset(scores.keys()):
        return _zero_ranking_metrics()

    sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_items = _select_top_k_items(
        sorted_items=sorted_items,
        k=k,
        adjust_for_ties=adjust_for_ties,
    )

    top_vars = {var for var, _ in top_items}

    true_positive = len(top_vars & root_cause_set)
    false_positive = len(top_vars - root_cause_set)
    false_negative = len(root_cause_set - top_vars)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    ndcg = _binary_ndcg(top_items, root_cause_set)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ndcg": float(ndcg),
    }


def _select_top_k_items(
    sorted_items: list[tuple[str, float]],
    k: int,
    adjust_for_ties: bool,
) -> list[tuple[str, float]]:
    """Select top-k scored items with optional tie adjustment."""
    if k >= len(sorted_items):
        return sorted_items

    kth_score = sorted_items[k - 1][1]

    if adjust_for_ties:
        return [
            (var, score)
            for var, score in sorted_items
            if score >= kth_score
        ]

    higher_items = [
        (var, score)
        for var, score in sorted_items
        if score > kth_score
    ]

    tied_items = [
        (var, score)
        for var, score in sorted_items
        if score == kth_score
    ]

    remaining = k - len(higher_items)

    if remaining <= 0:
        return higher_items[:k]

    sampled_ties = random.sample(tied_items, min(remaining, len(tied_items)))
    return higher_items + sampled_ties


def _binary_ndcg(
    ranked_items: list[tuple[str, float]],
    root_cause_set: set[str],
) -> float:
    """Compute NDCG with binary relevance."""
    dcg = 0.0

    for rank, (var, _) in enumerate(ranked_items):
        if var in root_cause_set:
            dcg += 1.0 / math.log2(rank + 2)

    ideal_hits = min(len(root_cause_set), len(ranked_items))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def _zero_ranking_metrics() -> dict[str, float]:
    """Return zero ranking metrics."""
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "ndcg": 0.0,
    }


def _failed_algorithm_metrics() -> dict[str, float]:
    """
    Metrics used when an algorithm completely fails for an experiment.

    Use 0.0 instead of np.nan so ordinary np.mean-based aggregation does not
    propagate NaNs.
    """
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "ndcg": 0.0,
        "skipped_samples": 0.0,
        "evaluated_samples": 0.0,
        "failure_rate": 1.0,
    }


# =============================================================================
# Dataset helpers
# =============================================================================

def subsample_dataset(
    df: pd.DataFrame,
    n: int,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Subsample a dataframe without replacement."""
    if n > len(df):
        raise ValueError(
            f"n={n} is larger than the number of rows in the dataframe "
            f"({len(df)})."
        )

    return df.sample(
        n=n,
        replace=False,
        random_state=random_state,
    ).reset_index(drop=True)


def _iter_anomaly_samples(
    anomaly_sample: pd.DataFrame,
    max_samples: int | None = None,
) -> Iterable[pd.DataFrame]:
    """
    Iterate over anomaly samples one row at a time.

    If max_samples is provided, only the first max_samples rows are used after
    subsampling.
    """
    if max_samples is not None:
        anomaly_sample = subsample_dataset(
            anomaly_sample,
            n=min(max_samples, len(anomaly_sample)),
        )

    for i in range(anomaly_sample.shape[0]):
        yield anomaly_sample.iloc[i:i + 1]


# =============================================================================
# Error handling
# =============================================================================

def is_cholesky_skippable_error(error: Exception) -> bool:
    """
    Return True for known numerical failures from Cholesky/RCD's LassoCV
    dimension-reduction step.

    This is intentionally narrow so real bugs are not silently hidden.
    """
    msg = str(error)

    return (
        "Gram matrix passed in via 'precompute' parameter did not pass validation"
        in msg
        or ("Gram matrix" in msg and "precompute" in msg)
    )


# =============================================================================
# Generic evaluation helpers
# =============================================================================

def _evaluate_scores(
    score_fn: Callable[[pd.DataFrame], dict[str, float]],
    experiment_data: dict[str, Any],
    k: int,
    adjust_for_ties: bool,
) -> tuple[dict[str, float], float]:
    """
    Run a scoring function once and compute ranking metrics.

    score_fn receives anomaly_data and returns a score dictionary.
    """
    start_time = time.perf_counter()

    scores = score_fn(experiment_data["anomaly_sample"])

    elapsed_time = time.perf_counter() - start_time

    metrics = top_k_metrics(
        scores=scores,
        root_causes=_as_root_cause_list(experiment_data["root_cause"]),
        k=k,
        adjust_for_ties=adjust_for_ties,
    )

    return metrics, elapsed_time


def _evaluate_scores_in_batch(
    score_fn: Callable[[pd.DataFrame], dict[str, float]],
    experiment_data: dict[str, Any],
    k: int,
    adjust_for_ties: bool,
    aggregate_method: str = "mean",
    desc: str = "Batch evaluation",
    max_samples: int | None = None,
) -> tuple[dict[str, float], float]:
    """
    Evaluate a scoring function on anomaly samples one by one and aggregate
    scores across samples.
    """
    start_time = time.perf_counter()
    score_dicts = []

    for anomaly_data in tqdm(
        _iter_anomaly_samples(
            experiment_data["anomaly_sample"],
            max_samples=max_samples,
        ),
        desc=desc,
    ):
        scores = score_fn(anomaly_data)

        if scores is not None and len(scores) > 0:
            score_dicts.append(scores)

    elapsed_time = time.perf_counter() - start_time

    if len(score_dicts) == 0:
        return _failed_algorithm_metrics(), elapsed_time

    aggregated_scores = _aggregate_score_dicts(
        score_dicts=score_dicts,
        aggregate_method=aggregate_method,
    )

    metrics = top_k_metrics(
        scores=aggregated_scores,
        root_causes=_as_root_cause_list(experiment_data["root_cause"]),
        k=k,
        adjust_for_ties=adjust_for_ties,
    )

    return metrics, elapsed_time


def _aggregate_score_dicts(
    score_dicts: list[dict[str, float]],
    aggregate_method: str,
) -> dict[str, float]:
    """
    Aggregate a list of score dictionaries.

    Missing values are filled with 0.0 to avoid KeyErrors when some algorithms
    only score a subset of nodes.
    """
    if len(score_dicts) == 0:
        return {}

    keys = _score_key_order(score_dicts)

    score_matrix = np.array(
        [
            [float(score_dict.get(key, 0.0)) for key in keys]
            for score_dict in score_dicts
        ],
        dtype=float,
    )

    if aggregate_method == "mean":
        aggregated = np.mean(score_matrix, axis=0)
    elif aggregate_method == "max":
        aggregated = np.max(score_matrix, axis=0)
    else:
        raise ValueError(
            f"Unknown aggregate_method={aggregate_method}. "
            "Expected one of ['mean', 'max']."
        )

    return {
        key: float(value)
        for key, value in zip(keys, aggregated)
    }


def _score_key_order(score_dicts: list[dict[str, float]]) -> list[str]:
    """
    Get a stable key order from a list of score dictionaries.

    Keys from the first score dictionary are kept first. New keys discovered in
    later dictionaries are appended in sorted order.
    """
    first_keys = list(score_dicts[0].keys())
    seen = set(first_keys)
    extra_keys = set()

    for score_dict in score_dicts[1:]:
        for key in score_dict:
            if key not in seen:
                extra_keys.add(key)

    return first_keys + sorted(extra_keys)


def _store_method_results(
    results: dict[str, Any],
    prefix: str,
    metrics: dict[str, float],
    elapsed_time: float,
) -> None:
    """Store ranking metrics and elapsed time under method-prefixed keys."""
    for key, value in metrics.items():
        results[f"{prefix}_{key}"] = value

    results[f"{prefix}_time"] = elapsed_time


# =============================================================================
# Individual method evaluators
# =============================================================================

def evaluate_score_ordering(
    graph,
    experiment_data: dict[str, Any],
    k: int,
    adjust_for_ties: bool,
    batch: bool = False,
    aggregate_method: str = "mean",
) -> tuple[dict[str, float], float]:
    def score_fn(anomaly_data: pd.DataFrame) -> dict[str, float]:
        return apply_score_ordering(
            graph=graph,
            normal_data=experiment_data["training_sample"],
            anomaly_data=anomaly_data,
            anomaly_scorer=RescaledMedianCDFQuantileScorer,
        )

    if batch:
        return _evaluate_scores_in_batch(
            score_fn=score_fn,
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
            aggregate_method=aggregate_method,
            desc="Score Ordering",
        )

    return _evaluate_scores(score_fn, experiment_data, k, adjust_for_ties)


def evaluate_traversal(
    graph,
    experiment_data: dict[str, Any],
    k: int,
    adjust_for_ties: bool,
    batch: bool = False,
    aggregate_method: str = "mean",
) -> tuple[dict[str, float], float]:
    def score_fn(anomaly_data: pd.DataFrame) -> dict[str, float]:
        return apply_simple_traversal(
            graph=graph,
            target_node=experiment_data["target_node"],
            normal_data=experiment_data["training_sample"],
            anomaly_data=anomaly_data,
            anomaly_scorer=RescaledMedianCDFQuantileScorer,
            anomaly_threshold=3.0,
            debug=False,
        )

    if batch:
        return _evaluate_scores_in_batch(
            score_fn=score_fn,
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
            aggregate_method=aggregate_method,
            desc="Traversal",
        )

    return _evaluate_scores(score_fn, experiment_data, k, adjust_for_ties)


def evaluate_smooth_traversal(
    graph,
    experiment_data: dict[str, Any],
    k: int,
    adjust_for_ties: bool,
    batch: bool = False,
    aggregate_method: str = "mean",
) -> tuple[dict[str, float], float]:
    def score_fn(anomaly_data: pd.DataFrame) -> dict[str, float]:
        return apply_smooth_traversal(
            graph=graph,
            target_node=experiment_data["target_node"],
            normal_data=experiment_data["training_sample"],
            anomaly_data=anomaly_data,
            anomaly_scorer=RescaledMedianCDFQuantileScorer,
            debug=False,
        )

    if batch:
        return _evaluate_scores_in_batch(
            score_fn=score_fn,
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
            aggregate_method=aggregate_method,
            desc="Smooth Traversal",
        )

    return _evaluate_scores(score_fn, experiment_data, k, adjust_for_ties)


def evaluate_cholesky(
    experiment_data: dict[str, Any],
    k: int,
    adjust_for_ties: bool,
    batch: bool = False,
    aggregate_method: str = "mean",
    cholesky_type: str = "highdim",
) -> tuple[dict[str, float], float]:
    def score_fn(anomaly_data: pd.DataFrame) -> dict[str, float]:
        return apply_cholesky(
            normal_data=experiment_data["training_sample"],
            anomaly_data=anomaly_data,
            cholesky_type=cholesky_type,
        )

    if batch:
        return _evaluate_cholesky_batch(
            score_fn=score_fn,
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
            aggregate_method=aggregate_method,
        )

    start_time = time.perf_counter()

    try:
        scores = score_fn(experiment_data["anomaly_sample"])
    except Exception as error:
        if is_cholesky_skippable_error(error):
            print("[Warning] Cholesky failed and was skipped for this experiment.")
            print(f"[Warning] Error message: {error}")

            elapsed_time = time.perf_counter() - start_time
            return _failed_algorithm_metrics(), elapsed_time

        raise

    elapsed_time = time.perf_counter() - start_time

    metrics = top_k_metrics(
        scores=scores,
        root_causes=_as_root_cause_list(experiment_data["root_cause"]),
        k=k,
        adjust_for_ties=adjust_for_ties,
    )

    return metrics, elapsed_time


def _evaluate_cholesky_batch(
    score_fn: Callable[[pd.DataFrame], dict[str, float]],
    experiment_data: dict[str, Any],
    k: int,
    adjust_for_ties: bool,
    aggregate_method: str,
) -> tuple[dict[str, float], float]:
    start_time = time.perf_counter()

    score_dicts = []
    skipped_samples = 0
    total_samples = experiment_data["anomaly_sample"].shape[0]

    for sample_idx, anomaly_data in enumerate(
        tqdm(
            _iter_anomaly_samples(experiment_data["anomaly_sample"]),
            desc="Cholesky",
        )
    ):
        try:
            scores = score_fn(anomaly_data)
        except Exception as error:
            if is_cholesky_skippable_error(error):
                skipped_samples += 1
                tqdm.write(
                    f"[Warning] Skipping Cholesky anomaly sample {sample_idx} "
                    "because of a known Gram-matrix error."
                )
                tqdm.write(f"[Warning] Error message: {error}")
                continue

            raise

        if scores is None or len(scores) == 0:
            skipped_samples += 1
            tqdm.write(
                f"[Warning] Skipping Cholesky anomaly sample {sample_idx} "
                "because scores are empty."
            )
            continue

        score_dicts.append(scores)

    elapsed_time = time.perf_counter() - start_time

    if len(score_dicts) == 0:
        metrics = _failed_algorithm_metrics()
        return metrics, elapsed_time

    aggregated_scores = _aggregate_score_dicts(
        score_dicts=score_dicts,
        aggregate_method=aggregate_method,
    )

    metrics = top_k_metrics(
        scores=aggregated_scores,
        root_causes=_as_root_cause_list(experiment_data["root_cause"]),
        k=k,
        adjust_for_ties=adjust_for_ties,
    )

    metrics["skipped_samples"] = float(skipped_samples)
    metrics["evaluated_samples"] = float(len(score_dicts))
    metrics["failure_rate"] = float(skipped_samples / max(total_samples, 1))

    return metrics, elapsed_time


def evaluate_baro(
    experiment_data: dict[str, Any],
    k: int,
    adjust_for_ties: bool,
) -> tuple[dict[str, float], float]:
    def score_fn(anomaly_data: pd.DataFrame) -> dict[str, float]:
        return apply_baro(
            normal_data=experiment_data["training_sample"],
            anomaly_data=anomaly_data,
        )

    return _evaluate_scores(score_fn, experiment_data, k, adjust_for_ties)


def evaluate_rcg_0(
    graph,
    experiment_data: dict[str, Any],
    k: int,
    adjust_for_ties: bool,
    batch: bool = False,
    aggregate_method: str = "mean",
) -> tuple[dict[str, float], float]:
    def score_fn(anomaly_data: pd.DataFrame) -> dict[str, float]:
        return apply_rcg_0(
            graph=graph,
            normal_data=experiment_data["training_sample"],
            anomaly_data=anomaly_data,
        )

    if batch:
        return _evaluate_scores_in_batch(
            score_fn=score_fn,
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
            aggregate_method=aggregate_method,
            desc="RCG-0",
            max_samples=1,
        )

    return _evaluate_scores(score_fn, experiment_data, k, adjust_for_ties)


def evaluate_counterfactual_contribution(
    graph,
    experiment_data: dict[str, Any],
    k: int,
    adjust_for_ties: bool,
    batch: bool = False,
    aggregate_method: str = "mean",
) -> tuple[dict[str, float], float]:
    def score_fn(anomaly_data: pd.DataFrame) -> dict[str, float]:
        return apply_counterfactual_contribution(
            graph=graph,
            target_node=experiment_data["target_node"],
            normal_data=experiment_data["training_sample"],
            anomaly_data=anomaly_data,
        )

    start_time = time.perf_counter()

    try:
        if batch:
            return _evaluate_scores_in_batch(
                score_fn=score_fn,
                experiment_data=experiment_data,
                k=k,
                adjust_for_ties=adjust_for_ties,
                aggregate_method=aggregate_method,
                desc="Counterfactual Contribution",
                max_samples=1,
            )

        return _evaluate_scores(score_fn, experiment_data, k, adjust_for_ties)

    except Exception as error:
        if is_cholesky_skippable_error(error):
            print(
                "[Warning] Counterfactual contribution failed and was skipped "
                "for this experiment."
            )
            print(f"[Warning] Error message: {error}")

            elapsed_time = time.perf_counter() - start_time
            return _failed_algorithm_metrics(), elapsed_time

        raise


def evaluate_circa(
    graph,
    experiment_data: dict[str, Any],
    k: int,
    adjust_for_ties: bool,
) -> tuple[dict[str, float], float]:
    def score_fn(anomaly_data: pd.DataFrame) -> dict[str, float]:
        return apply_circa(
            graph=graph.reverse(),
            target_node=experiment_data["target_node"],
            normal_data=experiment_data["training_sample"],
            anomaly_data=anomaly_data,
        )

    return _evaluate_scores(score_fn, experiment_data, k, adjust_for_ties)


def evaluate_rcd(
    graph,
    experiment_data: dict[str, Any],
    k: int,
    adjust_for_ties: bool,
) -> tuple[dict[str, float], float]:
    def score_fn(anomaly_data: pd.DataFrame) -> dict[str, float]:
        return apply_rcd(
            graph=graph.reverse(),
            target_node=experiment_data["target_node"],
            normal_data=experiment_data["training_sample"],
            anomaly_data=anomaly_data,
        )

    return _evaluate_scores(score_fn, experiment_data, k, adjust_for_ties)


def evaluate_epsilon_diagnosis(
    graph,
    experiment_data: dict[str, Any],
    k: int,
    adjust_for_ties: bool,
) -> tuple[dict[str, float], float]:
    def score_fn(anomaly_data: pd.DataFrame) -> dict[str, float]:
        return apply_epsilon_diagnosis(
            graph=graph.reverse(),
            target_node=experiment_data["target_node"],
            normal_data=experiment_data["training_sample"],
            anomaly_data=anomaly_data,
        )

    return _evaluate_scores(score_fn, experiment_data, k, adjust_for_ties)


def evaluate_stable_rca(
    args: Any,
    graph,
    experiment_data: dict[str, Any],
    k: int,
    adjust_for_ties: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Evaluate StableRCA.

    apply_stable_rca returns:
        scores, runtime_info
    """
    scores, runtime_info = apply_stable_rca(
        args,
        graph=graph,
        root_cause=experiment_data["root_cause"],
        normal_data=experiment_data["training_sample"],
        anomaly_data=experiment_data["anomaly_sample"],
    )

    metrics = top_k_metrics(
        scores=scores,
        root_causes=_as_root_cause_list(experiment_data["root_cause"]),
        k=k,
        adjust_for_ties=adjust_for_ties,
    )

    runtime_metrics = _stable_rca_runtime_metrics(runtime_info)

    return metrics, runtime_metrics


def _stable_rca_runtime_metrics(
    runtime_info: dict[str, Any],
) -> dict[str, float]:
    """Extract StableRCA runtime and count metrics."""
    runtime_metrics = {
        "stable_rca_time": float(runtime_info.get("total_time", 0.0)),
        "stable_rca_phase_1_mds_time": float(
            runtime_info.get("phase_1_mds_time", 0.0)
        ),
        "stable_rca_phase_2_mb_total_time": float(
            runtime_info.get("phase_2_mb_total_time", 0.0)
        ),
        "stable_rca_phase_3_cds_total_time": float(
            runtime_info.get("phase_3_cds_total_time", 0.0)
        ),
        "stable_rca_continuous_total_time": float(
            runtime_info.get("continuous_total_time", 0.0)
        ),
        "stable_rca_discrete_total_time": float(
            runtime_info.get("discrete_total_time", 0.0)
        ),
        "stable_rca_n_shift_nodes": float(
            runtime_info.get("n_shift_nodes", 0.0)
        ),
        "stable_rca_n_continuous_shift_nodes": float(
            runtime_info.get("n_continuous_shift_nodes", 0.0)
        ),
        "stable_rca_n_discrete_shift_nodes": float(
            runtime_info.get("n_discrete_shift_nodes", 0.0)
        ),
    }

    per_node = runtime_info.get("per_node", {})

    if len(per_node) == 0:
        runtime_metrics.update(
            {
                "stable_rca_avg_phase_2_mb_time_per_shift_node": 0.0,
                "stable_rca_avg_phase_3_cds_time_per_shift_node": 0.0,
                "stable_rca_avg_total_time_per_shift_node": 0.0,
            }
        )
        return runtime_metrics

    runtime_metrics.update(
        {
            "stable_rca_avg_phase_2_mb_time_per_shift_node": float(
                np.mean(
                    [
                        node_info.get("phase_2_mb_time", 0.0)
                        for node_info in per_node.values()
                    ]
                )
            ),
            "stable_rca_avg_phase_3_cds_time_per_shift_node": float(
                np.mean(
                    [
                        node_info.get("phase_3_cds_time", 0.0)
                        for node_info in per_node.values()
                    ]
                )
            ),
            "stable_rca_avg_total_time_per_shift_node": float(
                np.mean(
                    [
                        node_info.get("total_node_time", 0.0)
                        for node_info in per_node.values()
                    ]
                )
            ),
        }
    )

    return runtime_metrics


# =============================================================================
# Main multi-method evaluator
# =============================================================================

def evaluate_algorithms(
    experiment_data: dict[str, Any],
    args: Any,
    methods: list[str],
    k: int = 1,
    adjust_for_ties: bool = False,
    batch: bool = False,
    _aggregate_method: str = "mean",
) -> dict[str, Any]:
    """
    Evaluate selected RCA algorithms and return method-prefixed metrics.

    Returns
    -------
    results:
        Example keys:
        - score_ordering_precision
        - score_ordering_recall
        - score_ordering_f1
        - score_ordering_ndcg
        - score_ordering_time
    """
    graph = experiment_data["graph"]
    results: dict[str, Any] = {}

    if "score_ordering" in methods:
        metrics, elapsed_time = evaluate_score_ordering(
            graph=graph,
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
            batch=batch,
            aggregate_method=_aggregate_method,
        )
        _store_method_results(results, "score_ordering", metrics, elapsed_time)

    if "traversal" in methods:
        metrics, elapsed_time = evaluate_traversal(
            graph=graph,
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
            batch=batch,
            aggregate_method=_aggregate_method,
        )
        _store_method_results(results, "traversal", metrics, elapsed_time)

    if "smooth_traversal" in methods:
        metrics, elapsed_time = evaluate_smooth_traversal(
            graph=graph,
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
            batch=batch,
            aggregate_method=_aggregate_method,
        )
        _store_method_results(results, "smooth_traversal", metrics, elapsed_time)

    if "cholesky" in methods:
        metrics, elapsed_time = evaluate_cholesky(
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
            batch=batch,
            aggregate_method=_aggregate_method,
        )
        _store_method_results(results, "cholesky", metrics, elapsed_time)

    if "baro" in methods:
        metrics, elapsed_time = evaluate_baro(
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
        )
        _store_method_results(results, "baro", metrics, elapsed_time)

    if "rcg_0" in methods:
        metrics, elapsed_time = evaluate_rcg_0(
            graph=graph,
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
            batch=batch,
            aggregate_method=_aggregate_method,
        )
        _store_method_results(results, "rcg_0", metrics, elapsed_time)

    if "counterfactual" in methods:
        metrics, elapsed_time = evaluate_counterfactual_contribution(
            graph=graph,
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
            batch=batch,
            aggregate_method=_aggregate_method,
        )
        _store_method_results(
            results,
            "counterfactual_contribution",
            metrics,
            elapsed_time,
        )

    if "circa" in methods:
        metrics, elapsed_time = evaluate_circa(
            graph=graph,
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
        )
        _store_method_results(results, "circa", metrics, elapsed_time)

    if "rcd" in methods:
        metrics, elapsed_time = evaluate_rcd(
            graph=graph,
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
        )
        _store_method_results(results, "rcd", metrics, elapsed_time)

    if "epsilon_diagnosis" in methods:
        metrics, elapsed_time = evaluate_epsilon_diagnosis(
            graph=graph,
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
        )
        _store_method_results(results, "epsilon_diagnosis", metrics, elapsed_time)

    if "stable_rca" in methods:
        metrics, runtime_metrics = evaluate_stable_rca(
            args=args,
            graph=graph,
            experiment_data=experiment_data,
            k=k,
            adjust_for_ties=adjust_for_ties,
        )

        for key, value in metrics.items():
            results[f"stable_rca_{key}"] = value

        results.update(runtime_metrics)

    return results