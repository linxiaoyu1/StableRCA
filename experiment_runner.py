# ./experiment_runner.py
from __future__ import annotations

import os
import random
from typing import Any, Callable

import pandas as pd
from dowhy.gcm.util.general import set_random_seed
from tqdm import tqdm

from evaluate_algorithms import evaluate_algorithms
from result_saver import load_results

from data_simulators.prorca import pro_rca_data_generator
from data_simulators.sock_shop import preprocess_sock_shop
from data_simulators.synthetic_data import synthetic_data_generator
from data_simulators.causal_chamber import causal_chamber_generator
from data_simulators.causal_man import causal_man_generator
from data_simulators.rcaeval import (
    parse_rcaeval_case_dir,
    preprocess_rcaeval_case,
)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_SYNTHETIC_NOISE_TYPES = [
    "Gaussian",
    "Gumbel",
    "Uniform",
    "Exponential",
]

METHOD_TO_RESULT_PREFIX = {
    "score_ordering": "score_ordering",
    "smooth_traversal": "smooth_traversal",
    "traversal": "traversal",
    "cholesky": "cholesky",
    "circa": "circa",
    "counterfactual": "counterfactual_contribution",
    "counterfactual_contribution": "counterfactual_contribution",
    "rcd": "rcd",
    "epsilon_diagnosis": "epsilon_diagnosis",
    "baro": "baro",
    "rcg_0": "rcg_0",
    "stable_rca": "stable_rca",
}

BASE_METRICS = [
    "time",
    "precision",
    "recall",
    "f1",
    "ndcg",
]

STABLE_RCA_EXTRA_METRICS = [
    "stable_rca_phase_1_mds_time",
    "stable_rca_phase_2_mb_total_time",
    "stable_rca_phase_3_cds_total_time",
    "stable_rca_continuous_total_time",
    "stable_rca_discrete_total_time",
    "stable_rca_n_shift_nodes",
    "stable_rca_n_continuous_shift_nodes",
    "stable_rca_n_discrete_shift_nodes",
    "stable_rca_avg_phase_2_mb_time_per_shift_node",
    "stable_rca_avg_phase_3_cds_time_per_shift_node",
    "stable_rca_avg_total_time_per_shift_node",
]


# =============================================================================
# Main entry
# =============================================================================

def run_experiments(
    config: dict[str, Any],
    args: Any,
) -> dict[str, Any]:
    """
    Run RCA experiments according to config["experiment_mode"].

    The expected experiment_data dictionary returned by each data generator is:

        {
            "graph": ...,
            "training_sample": ...,
            "anomaly_sample": ...,
            "root_cause": ...,
            "target_node": ...,
        }
    """
    results = load_results(config["results_path"])

    if "config" not in results:
        results["config"] = dict(config)

    experiment_mode = config["experiment_mode"]

    if experiment_mode == "synthetic_data":
        _run_synthetic_data_experiments(results, config, args)

    elif experiment_mode == "pro_rca":
        _run_standard_generator_experiments(
            results=results,
            config=config,
            args=args,
            parameter_list=config["pro_rca_anomaly_list"],
            generator_fn=lambda anomaly_tuple: pro_rca_data_generator(
                config,
                anomaly_tuple=anomaly_tuple,
            ),
            desc="ProRCA",
        )

    elif experiment_mode == "sock_shop":
        _run_sock_shop_experiments(results, config, args)

    elif experiment_mode == "causal_chamber":
        _run_standard_generator_experiments(
            results=results,
            config=config,
            args=args,
            parameter_list=config["causal_chamber_list"],
            generator_fn=lambda intervention_name: causal_chamber_generator(
                intervention_name=intervention_name,
                config=config,
            ),
            desc="CausalChamber",
        )

    elif experiment_mode == "causal_man":
        _run_standard_generator_experiments(
            results=results,
            config=config,
            args=args,
            parameter_list=config["causal_man_list"],
            generator_fn=lambda data_setting: causal_man_generator(
                data_setting=data_setting,
                config=config,
            ),
            desc="CausalMan",
        )

    elif experiment_mode == "rcaeval":
        _run_rcaeval_experiments(results, config, args)

    else:
        raise ValueError(f"Unsupported experiment_mode: {experiment_mode}")

    return results


# =============================================================================
# Experiment runners
# =============================================================================

def _run_synthetic_data_experiments(
    results: dict[str, Any],
    config: dict[str, Any],
    args: Any,
) -> None:
    """
    Run synthetic SCM experiments.

    """
    parameter = "synthetic_data"
    results[parameter] = initialise_result_storage(config)

    with tqdm(total=config["number_trials"], desc="Synthetic data") as progress_bar:
        num_finished = 0
        seed_offset = 0

        while num_finished < config["number_trials"]:
            trial_seed = seed_offset
            seed_offset += 1

            set_random_seed(trial_seed)

            args.noise_type = random.choice(DEFAULT_SYNTHETIC_NOISE_TYPES)

            experiment_data = synthetic_data_generator(args)

            if experiment_data is None:
                continue

            result_metrics = _evaluate_experiment_data(
                experiment_data=experiment_data,
                config=config,
                args=args,
            )

            update_results(results[parameter], result_metrics)

            num_finished += 1
            progress_bar.update(1)


def _run_standard_generator_experiments(
    results: dict[str, Any],
    config: dict[str, Any],
    args: Any,
    parameter_list: list,
    generator_fn: Callable[[Any], dict[str, Any] | None],
    desc: str,
) -> None:
    """
    Run experiments for generators indexed by a parameter list.

    Used by:
    - ProRCA
    - CausalChamber
    - CausalMan
    """
    for parameter_index, parameter in enumerate(parameter_list):
        print(f"Parameter = {parameter}")

        results[parameter] = initialise_result_storage(config)

        with tqdm(
            total=config["number_trials"],
            desc=f"{desc}: {parameter}",
        ) as progress_bar:
            num_finished = 0
            seed_offset = 0

            while num_finished < config["number_trials"]:
                trial_seed = parameter_index * config["number_trials"] + seed_offset
                seed_offset += 1

                set_random_seed(trial_seed)

                experiment_data = generator_fn(parameter)

                if experiment_data is None:
                    continue

                result_metrics = _evaluate_experiment_data(
                    experiment_data=experiment_data,
                    config=config,
                    args=args,
                )

                update_results(results[parameter], result_metrics)

                num_finished += 1
                progress_bar.update(1)


def _run_sock_shop_experiments(
    results: dict[str, Any],
    config: dict[str, Any],
    args: Any,
) -> None:
    """
    Run Sock-Shop experiments.
    """
    sock_shop_path = config["sock_shop_data_path"]

    if not os.path.exists(sock_shop_path):
        raise FileNotFoundError(
            f"Base path not found: {sock_shop_path}\n"
            "If you have not downloaded the Sock-Shop dataset, run the "
            "'download_sock_shop.py' script first."
        )

    for issue_type in config["sock_shop_list"]:
        print(f"Parameter = {issue_type}")

        results[issue_type] = initialise_result_storage(config)

        issue_dirs = [
            directory
            for directory in os.listdir(sock_shop_path)
            if directory.endswith(f"_{issue_type}")
            and os.path.isdir(os.path.join(sock_shop_path, directory))
        ]

        for service_dir in issue_dirs:
            service_path = os.path.join(sock_shop_path, service_dir)
            root_cause = os.path.basename(service_path.rstrip("/")).rsplit("_", 1)[0]

            for replicate in range(1, 6):
                replicate_path = os.path.join(service_path, str(replicate))
                csv_path = os.path.join(replicate_path, "simple_data.csv")
                inject_time_path = os.path.join(replicate_path, "inject_time.txt")

                if not os.path.exists(csv_path):
                    print(f"Missing file: {csv_path}")
                    continue

                if not os.path.exists(inject_time_path):
                    print(f"Missing file: {inject_time_path}")
                    continue

                raw_data = pd.read_csv(csv_path)
                print(f"Loaded {csv_path}")

                with open(inject_time_path, "r", encoding="utf-8") as file:
                    inject_time = int(file.readline().strip())

                experiment_data = preprocess_sock_shop(
                    raw_data,
                    root_cause,
                    issue_type,
                    inject_time,
                    use_all_anomaly_samples=config["use_all_sock_shop_anomaly_samples"],
                )

                result_metrics = _evaluate_experiment_data(
                    experiment_data=experiment_data,
                    config=config,
                    args=args,
                )

                update_results(results[issue_type], result_metrics)


def _run_rcaeval_experiments(
    results: dict[str, Any],
    config: dict[str, Any],
    args: Any,
) -> None:
    """
    Run RCAEval experiments.
    """
    base_path = config["rcaeval_data_path"]

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path not found: {base_path}")

    case_dirs = _discover_rcaeval_case_dirs(base_path)

    parameter = "all_cases"
    results[parameter] = initialise_result_storage(config)

    total_runs = len(case_dirs) * config["number_trials"]

    with tqdm(total=total_runs, desc="RCAEval cases") as progress_bar:
        for case_index, case_dir in enumerate(case_dirs):
            case_name = os.path.basename(case_dir)

            dataset_prefix, root_cause, fault_label, instance_id = parse_rcaeval_case_dir(
                case_name
            )

            for trial in range(config["number_trials"]):
                trial_seed = case_index * config["number_trials"] + trial
                set_random_seed(trial_seed)

                progress_bar.set_postfix_str(
                    f"{case_name} | trial {trial + 1}/{config['number_trials']}"
                )

                metadata = {
                    "case_name": case_name,
                    "dataset_prefix": dataset_prefix,
                    "root_cause": root_cause,
                    "fault_label": fault_label,
                    "instance_id": instance_id,
                    "trial": trial,
                }

                try:
                    experiment_data = preprocess_rcaeval_case(
                        case_dir=case_dir,
                        graph_builder=config.get("rcaeval_graph_builder", None),
                        use_all_anomaly_samples=config.get(
                            "use_all_rcaeval_anomaly_samples",
                            False,
                        ),
                        anomaly_threshold=config.get(
                            "rcaeval_anomaly_threshold",
                            3.0,
                        ),
                        window_size=config.get("rcaeval_window_size", None),
                        tdelta=config.get("rcaeval_tdelta", 0),
                    )

                    result_metrics = _evaluate_experiment_data(
                        experiment_data=experiment_data,
                        config=config,
                        args=args,
                    )

                    update_results(
                        results[parameter],
                        result_metrics,
                        metadata={**metadata, "status": "ok"},
                    )

                except Exception as exc:
                    print(f"\n[Skip] case={case_name}, trial={trial}, error={exc}")

                    results[parameter].setdefault("case_results", [])
                    results[parameter]["case_results"].append(
                        {
                            **metadata,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )

                progress_bar.update(1)


# =============================================================================
# Evaluation and result storage
# =============================================================================

def _evaluate_experiment_data(
    experiment_data: dict[str, Any],
    config: dict[str, Any],
    args: Any,
) -> dict[str, Any]:
    """
    Evaluate selected algorithms on one RCA benchmark instance.
    """
    return evaluate_algorithms(
        experiment_data,
        args=args,
        methods=config["methods"],
        k=config["k"],
        adjust_for_ties=config["adjust_for_ties"],
        batch=getattr(args, "batch_mode", False),
        _aggregate_method=getattr(args, "aggregate_method", "mean"),
    )


def initialise_result_storage(config: dict[str, Any]) -> dict[str, list]:
    """
    Initialize metric storage for selected methods.
    """
    result: dict[str, list] = {}

    for method in config["methods"]:
        prefix = METHOD_TO_RESULT_PREFIX.get(method, method)

        for metric in BASE_METRICS:
            result[f"{prefix}_{metric}"] = []

    if "stable_rca" in config["methods"]:
        for metric_name in STABLE_RCA_EXTRA_METRICS:
            result[metric_name] = []

    return result


def update_results(
    results: dict[str, Any],
    metrics: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Append metrics to result storage.

    If metadata is provided, also append a case-level record to
    results["case_results"].
    """
    for key, value in metrics.items():
        results.setdefault(key, []).append(value)

    if metadata is not None:
        results.setdefault("case_results", [])
        results["case_results"].append({**metadata, **metrics})


# =============================================================================
# Data helpers
# =============================================================================

def _discover_rcaeval_case_dirs(base_path: str) -> list[str]:
    """
    Discover valid RCAEval case directories.
    """
    case_dirs = [
        os.path.join(base_path, directory)
        for directory in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, directory))
        and os.path.exists(os.path.join(base_path, directory, "metrics.json"))
        and os.path.exists(os.path.join(base_path, directory, "inject_time.txt"))
    ]

    return sorted(case_dirs)