# ./main.py

from __future__ import annotations

import argparse
import json
import os

from experiment_runner import run_experiments
from result_saver import save_results


def _parse_json_dict(value: str) -> dict:
    """Parse a JSON dictionary from command line."""
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON dictionary.")
    return parsed


def parse_config_from_args():
    parser = argparse.ArgumentParser(
        description="Root Cause Analysis Experiment Configuration"
    )

    # =========================================================================
    # Global experiment args
    # =========================================================================

    parser.add_argument(
        "--experiment_mode",
        type=str,
        choices=[
            "synthetic_data",
            "pro_rca",
            "sock_shop",
            "causal_chamber",
            "causal_man",
            "rcaeval",
        ],
        default="synthetic_data",
        help="Type of experiment.",
    )

    parser.add_argument(
        "--methods",
        type=str,
        default=(
            "score_ordering,smooth_traversal,traversal,cholesky,circa,"
            "counterfactual,rcd,epsilon_diagnosis,baro,rcg_0,stable_rca"
        ),
        help=(
            "Comma-separated list of methods to evaluate. Options include "
            "score_ordering, smooth_traversal, traversal, counterfactual, circa, "
            "cholesky, rcd, epsilon_diagnosis, baro, rcg_0, and stable_rca."
        ),
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Whether to run experiments in parallel.",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs for parallel running.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of top-k root causes to evaluate.",
    )

    parser.add_argument(
        "--number_trials",
        type=int,
        default=20,
        help="Number of graphs/data samples per experiment setting.",
    )

    parser.add_argument(
        "--adjust_for_ties",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to account for ties in top-k evaluation.",
    )

    parser.add_argument(
        "--results_path",
        type=str,
        default="./results/results.npy",
        help="Path to save experiment results.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed.",
    )

    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to print intermediate results.",
    )

    # =========================================================================
    # Dataset args
    # =========================================================================

    parser.add_argument(
        "--use_all_sock_shop_anomaly_samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use all Sock-Shop anomaly-period samples.",
    )

    parser.add_argument(
        "--sock_shop_data_path",
        type=str,
        default="./datasets/sock-shop-2/",
        help="Path to the Sock-Shop data directory.",
    )

    parser.add_argument(
        "--causal_chamber_data_path",
        type=str,
        default="./datasets/causalchamber",
        help="Path to the CausalChamber data directory.",
    )

    parser.add_argument(
        "--causal_man_data_path",
        type=str,
        default="./datasets/causalman",
        help="Path to the CausalMan data directory.",
    )

    parser.add_argument(
        "--rcaeval_data_path",
        type=str,
        default="./datasets/RCAEval_v2",
        help="Path to the RCAEval case directory root.",
    )

    parser.add_argument(
        "--rcaeval_window_size",
        type=int,
        default=None,
        help="Optional number of rows kept before and after inject_time.",
    )

    parser.add_argument(
        "--rcaeval_tdelta",
        type=int,
        default=0,
        help="Offset added to inject_time to simulate delayed anomaly detection.",
    )

    parser.add_argument(
        "--use_all_rcaeval_anomaly_samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to keep all anomaly-period samples for RCAEval.",
    )

    parser.add_argument(
        "--rcaeval_anomaly_threshold",
        type=float,
        default=3.0,
        help="Threshold for selecting the first detected RCAEval anomaly sample.",
    )

    # =========================================================================
    # Synthetic SCM generator args
    # =========================================================================

    parser.add_argument(
        "--n_nodes",
        type=int,
        default=50,
        help="Number of nodes in the synthetic data generator.",
    )

    parser.add_argument(
        "--n_edges",
        type=int,
        default=100,
        help="Number of edges in the synthetic data generator.",
    )

    parser.add_argument(
        "--edge_multiplier",
        type=float,
        default=None,
        help=(
            "Optional shortcut: set n_edges = round(n_nodes * edge_multiplier). "
            "If omitted, n_edges is used directly."
        ),
    )

    parser.add_argument(
        "--graph_type",
        type=str,
        choices=["ER", "SF"],
        default="ER",
        help="Graph generator type for synthetic SCMs.",
    )

    parser.add_argument(
        "--function_type",
        type=str,
        choices=["Linear", "MLP", "Hybrid", "auto"],
        default="Hybrid",
        help="SCM function type.",
    )

    parser.add_argument(
        "--hybrid_mlp_prob",
        type=float,
        default=0.5,
        help="Probability of sampling an MLP edge function when function_type='Hybrid'.",
    )

    parser.add_argument(
        "--function_params",
        type=_parse_json_dict,
        default={"uniform_l": 0.25, "bias": False},
        help="Edge function parameters as a JSON dictionary.",
    )

    parser.add_argument(
        "--noise_type",
        choices=["Gaussian", "Laplace", "Gumbel", "Uniform", "Exponential", "auto"],
        default="auto",
        help="Noise type.",
    )

    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.5,
        help="Noise standard deviation.",
    )

    parser.add_argument(
        "--aggregate_noise",
        choices=["additive"],
        default="additive",
        help="Noise aggregation form.",
    )

    parser.add_argument(
        "--min_root",
        type=float,
        default=0.0,
        help="Minimum value of root nodes.",
    )

    parser.add_argument(
        "--max_root",
        type=float,
        default=1.0,
        help="Maximum value of root nodes.",
    )

    parser.add_argument(
        "--n_sample_normal",
        type=int,
        default=2000,
        help="Number of normal samples generated per synthetic graph.",
    )

    parser.add_argument(
        "--n_sample_abnormal",
        type=int,
        default=200,
        help="Number of abnormal samples generated per synthetic graph.",
    )

    parser.add_argument(
        "--anomaly_probability",
        type=float,
        default=0.05,
        help="Anomaly probability threshold in the synthetic data generator.",
    )

    parser.add_argument(
        "--intervention_type",
        type=str,
        choices=["hard", "soft_function", "soft_noise", "soft_distribution"],
        default="soft_function",
        help="Type of intervention in the synthetic data generator.",
    )

    parser.add_argument(
        "--n_intervention_nodes",
        type=int,
        default=1,
        help="Number of intervention nodes in the synthetic data generator.",
    )

    parser.add_argument(
        "--intervention_function_params",
        type=_parse_json_dict,
        default={"uniform_l": 0.25, "uniform_u": 1},
        help="Intervention function parameters as a JSON dictionary.",
    )

    parser.add_argument(
        "--hard_intervention_magnitude",
        type=float,
        default=1.5,
        help="Magnitude multiplier for hard interventions.",
    )

    parser.add_argument(
        "--soft_distribution_magnitude",
        type=float,
        default=1.5,
        help="Mean-shift strength for soft_distribution interventions.",
    )

    parser.add_argument(
        "--soft_distribution_std_scale",
        type=float,
        default=0.3,
        help="Std scale for soft_distribution interventions.",
    )

    parser.add_argument(
        "--soft_distribution_min_std",
        type=float,
        default=1e-6,
        help="Minimum std for soft_distribution interventions.",
    )

    parser.add_argument(
        "--soft_distribution_shift_direction",
        type=str,
        choices=["positive", "negative", "random"],
        default="positive",
        help="Direction of the soft_distribution mean shift.",
    )

    parser.add_argument(
        "--intervention_node_mode",
        type=str,
        choices=["non_root", "root", "any"],
        default="non_root",
        help="Which nodes are allowed to be selected as intervention nodes.",
    )

    parser.add_argument(
        "--target_node_mode",
        type=str,
        choices=["leaf", "non_leaf", "random"],
        default="random",
        help="How to choose the target node from affected nodes.",
    )

    parser.add_argument(
        "--min_intervention_descendants",
        type=int,
        default=1,
        help="Minimum number of descendants required for an intervention node.",
    )

    parser.add_argument(
        "--max_intervention_descendants",
        type=int,
        default=15,
        help="Maximum number of descendants allowed. Use -1 to disable.",
    )

    parser.add_argument(
        "--max_generation_attempts",
        type=int,
        default=1000,
        help="Maximum attempts to generate a valid synthetic RCA instance.",
    )

    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="Whether to aggregate across anomaly samples.",
    )

    parser.add_argument(
        "--aggregate_method",
        type=str,
        choices=["mean", "max"],
        default="mean",
        help="Aggregation method used in batch mode.",
    )

    # =========================================================================
    # Benchmark graph args
    # =========================================================================

    parser.add_argument(
        "--graph_mode",
        type=str,
        choices=["true", "xges", "corrupted"],
        default="true",
        help=(
            "Graph supplied to graph-based RCA methods. "
            "'true' uses the ground-truth DAG, 'xges' uses an XGES-estimated DAG, "
            "and 'corrupted' uses a manually perturbed DAG."
        ),
    )

    parser.add_argument(
        "--corrupt_delete_frac",
        type=float,
        default=0.15,
        help="Fraction of true graph edges randomly deleted in corrupted graph mode.",
    )

    parser.add_argument(
        "--corrupt_reverse_frac",
        type=float,
        default=0.10,
        help="Fraction of true graph edges randomly reversed in corrupted graph mode.",
    )

    parser.add_argument(
        "--corrupt_add_frac",
        type=float,
        default=0.10,
        help="Fraction of false-positive edges added in corrupted graph mode.",
    )

    parser.add_argument(
        "--corrupt_graph_seed",
        type=int,
        default=None,
        help="Optional random seed for corrupted graph generation.",
    )

    # =========================================================================
    # StableRCA args
    # =========================================================================

    parser.add_argument(
        "--discrete_threshold",
        type=float,
        default=10,
        help="Unique-value threshold used to determine discrete variables.",
    )

    parser.add_argument(
        "--p_value_threshold",
        type=float,
        default=0.001,
        help="P-value threshold for marginal distribution shift tests.",
    )

    parser.add_argument(
        "--stable_weighting_algorithm",
        choices=["dwr", "srdo"],
        default="srdo",
        help="Stable-learning sample weighting algorithm.",
    )

    parser.add_argument(
        "--stable_prediction_model_continuous",
        choices=["stg", "linear", "catboost"],
        default="catboost",
        help="Stable-learning prediction model for continuous variables.",
    )

    parser.add_argument(
        "--stable_prediction_model_categorical",
        choices=["linear", "catboost"],
        default="catboost",
        help="Stable-learning prediction model for categorical variables.",
    )

    parser.add_argument(
        "--order",
        choices=[1, 2, 3],
        type=int,
        default=1,
        help="Order of the DWR sample weighting algorithm.",
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=20000,
        help="Number of DWR optimization steps.",
    )

    parser.add_argument(
        "--num_epoch",
        type=int,
        default=10,
        help="Number of stable-selection iterations.",
    )

    parser.add_argument(
        "--period_MA",
        type=int,
        default=3,
        help="Moving-average period for STG feature selection ratios.",
    )

    parser.add_argument(
        "--feature_selection_quantile",
        type=float,
        default=0.6,
        help="Quantile threshold for feature selection.",
    )

    parser.add_argument(
        "--min_num_selected_features",
        type=int,
        default=1,
        help="Minimum number of selected features.",
    )

    parser.add_argument(
        "--lam_STG",
        type=float,
        default=3,
        help="STG regularization parameter.",
    )

    parser.add_argument(
        "--sigma_STG",
        type=float,
        default=0.1,
        help="STG noise scale.",
    )

    parser.add_argument(
        "--stg_epochs",
        type=int,
        default=5000,
        help="Number of training epochs for STG.",
    )

    parser.add_argument(
        "--selection_weight_ratio",
        type=float,
        default=0.18,
        help="Relative feature-weight threshold for feature selection.",
    )

    parser.add_argument(
        "--rca_metric",
        type=str,
        default="mse",
        help="Prediction metric for root-cause scoring.",
    )

    parser.add_argument(
        "--rca_threshold",
        type=float,
        default=0.22,
        help="Performance-drop threshold for determining root causes.",
    )

    parser.add_argument(
        "--rca_reweighting",
        action="store_true",
        help="Whether to use sample reweighting against covariate shift.",
    )

    parser.add_argument(
        "--default_large_score",
        type=float,
        default=500.0,
        help="Default large score if classification is out of support.",
    )

    args = parser.parse_args()

    if args.edge_multiplier is not None:
        args.n_edges = int(round(args.n_nodes * args.edge_multiplier))

    config = vars(args)
    config["methods"] = [
        name.strip().replace("-", "_")
        for name in args.methods.split(",")
        if name.strip()
    ]

    config["pro_rca_anomaly_list"] = [
        ("ExcessiveDiscount", 0.2, "DISCOUNT", "Apparel"),
        ("FulfillmentSpike", 3, "FULFILLMENT_COST", "Beauty"),
        ("ReturnSurge", 10, "RETURN_COST", "Accessories"),
        ("ShippingDisruption", 5, "SHIPPING_REVENUE", "PersonalCare"),
    ]

    config["sock_shop_list"] = [
        "cpu",
        "delay",
        "disk",
        "loss",
        "mem",
    ]

    config["causal_chamber_list"] = [
        "uniform_red_mid",
        "uniform_green_mid",
        "uniform_blue_mid",
        "uniform_red_strong",
        "uniform_green_strong",
        "uniform_blue_strong",
        "uniform_pol_1_mid",
        "uniform_pol_2_mid",
        "uniform_pol_1_strong",
        "uniform_pol_2_strong",
        "uniform_l_11_mid",
        "uniform_l_12_mid",
        "uniform_l_21_mid",
        "uniform_l_22_mid",
        "uniform_l_31_mid",
        "uniform_l_32_mid",
    ]

    config["causal_man_list"] = [
        "small_do_PF_M1_T1_Force_16000",
        "small_do_PF_M1_T1_Force_30000",
        "small_do_PF_M1_T1_Force_17000", 
        "small_do_PF_M1_T1_Force_17000_var3000_soft", 
        "small_do_PF_M1_T1_Fmax_18500",
        "small_do_PF_M1_T1_Fmax_18500_var3000_soft",
        "small_do_PF_M1_T1_sgrad_20",
        "small_do_PF_M1_T1_sgrad_20_var4_soft",
        "medium_do_PF_M1_T1_Force_17000", 
        "medium_do_PF_M1_T1_Force_17000_var3000_soft",
        "medium_do_PF_M1_T1_Fmax_18500",
        "medium_do_PF_M1_T1_Fmax_18500_var3000_soft",
        "medium_do_PF_M1_T1_sgrad_20",
        "medium_do_PF_M1_T1_sgrad_20_var4_soft",
    ]

    return config, args


def main() -> None:
    config, args = parse_config_from_args()

    results = run_experiments(config, args)

    print("Raw results")
    print(results)

    results_path = config["results_path"]
    results_dir = os.path.dirname(results_path)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    save_results(results, results_path)


if __name__ == "__main__":
    main()