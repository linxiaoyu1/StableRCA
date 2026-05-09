# ./data_simulators/rcaeval.py

import json
import os
from typing import Callable, Optional

import networkx as nx
import numpy as np
import pandas as pd

from dowhy.gcm import ITAnomalyScorer, RescaledMedianCDFQuantileScorer
from dowhy.gcm.anomaly_scorer import AnomalyScorer


def parse_rcaeval_case_dir(case_dir_name: str):
    """
    Supports both patterns:

    RE1 / RE2:
        re1ob_adservice_cpu_1
        re2tt_ts-auth-service_delay_3

    RE3:
        re3ob_adservice_f3_1
        re3ss_front-end_f2_3
        re3tt_ts-route-service_f3_6

    Returns:
        dataset_prefix, root_cause, fault_label, instance_id
    """
    parts = case_dir_name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected RCAEval case dir format: {case_dir_name}")

    dataset_prefix = parts[0]
    instance_id = parts[-1]
    fault_label = parts[-2]
    root_cause = "_".join(parts[1:-2])

    return dataset_prefix, root_cause, fault_label, instance_id

def drop_degenerate_columns(normal_data: pd.DataFrame, anomaly_data: pd.DataFrame):
    """
    Drop columns that are constant / degenerate in either split.
    These columns can break chi-square tests or be useless for RCA.
    """
    common_cols = [c for c in normal_data.columns if c in anomaly_data.columns]
    keep_cols = []

    for c in common_cols:
        n_unique_normal = normal_data[c].nunique(dropna=False)
        n_unique_anomaly = anomaly_data[c].nunique(dropna=False)

        if n_unique_normal < 2 or n_unique_anomaly < 2:
            continue

        keep_cols.append(c)

    return normal_data[keep_cols].copy(), anomaly_data[keep_cols].copy()

def load_metrics_json_as_dataframe(metrics_json_path: str) -> pd.DataFrame:
    """
    RCAEval metrics.json format:
        {
            "metric_name": [[timestamp, value], [timestamp, value], ...],
            ...
        }

    Different metrics may not share exactly the same timestamps, so we outer-merge.
    """
    with open(metrics_json_path, "r") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(
            f"Expected top-level dict in {metrics_json_path}, got {type(obj).__name__}"
        )

    metric_dfs = []

    for metric_name, values in obj.items():
        if not isinstance(values, list):
            raise ValueError(
                f"Metric '{metric_name}' in {metrics_json_path} is not a list."
            )

        if len(values) == 0:
            continue

        if not all(isinstance(x, list) and len(x) == 2 for x in values):
            raise ValueError(
                f"Metric '{metric_name}' in {metrics_json_path} is not in [[time, value], ...] format."
            )

        metric_df = pd.DataFrame(values, columns=["time", metric_name])
        metric_df = metric_df.drop_duplicates(subset=["time"], keep="last")
        metric_dfs.append(metric_df)

    if not metric_dfs:
        raise ValueError(f"No metric series found in {metrics_json_path}")

    df = metric_dfs[0]
    for metric_df in metric_dfs[1:]:
        df = df.merge(metric_df, on="time", how="outer")

    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def infer_target_node(dataset_prefix: str, columns: pd.Index) -> str:
    dataset_prefix = dataset_prefix.lower()

    if "ob" in dataset_prefix:
        for cand in ["frontend", "frontend-proxy", "frontend_1"]:
            if cand in columns:
                return cand

    if "ss" in dataset_prefix:
        if "front-end" in columns:
            return "front-end"

    if "tt" in dataset_prefix:
        if "ts-ui-dashboard" in columns:
            return "ts-ui-dashboard"

    return columns[0]


def make_empty_graph(columns: pd.Index) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(columns)
    return g


def prune_graph_to_observed_nodes(graph: nx.DiGraph, observed_nodes) -> nx.DiGraph:
    g = graph.copy()
    observed_nodes = set(observed_nodes)

    nodes_to_remove = []
    edges_to_add = []

    for node in list(g.nodes()):
        if node not in observed_nodes:
            children = list(g.successors(node))
            parents = list(g.predecessors(node))

            if len(children) > 1 and len(parents) > 0:
                raise ValueError(f"{node} is an unmeasured confounder!")

            nodes_to_remove.append(node)
            for p in parents:
                for c in children:
                    edges_to_add.append((p, c))

    if nodes_to_remove:
        g.remove_nodes_from(nodes_to_remove)
        g.add_edges_from(edges_to_add)

    return g


def preprocess_rcaeval_case(
    case_dir: str,
    graph_builder: Optional[Callable[[], nx.DiGraph]] = None,
    anomaly_scorer: Callable[[], AnomalyScorer] = RescaledMedianCDFQuantileScorer,
    anomaly_threshold: float = 3.0,
    use_all_anomaly_samples: bool = False,
    window_size: Optional[int] = None,
    tdelta: int = 0,
):
    case_name = os.path.basename(case_dir)
    dataset_prefix, root_cause, fault_label, instance_id = parse_rcaeval_case_dir(case_name)

    inject_time_path = os.path.join(case_dir, "inject_time.txt")
    metrics_json_path = os.path.join(case_dir, "metrics.json")

    with open(inject_time_path, "r") as f:
        inject_time = int(f.readline().strip()) + tdelta

    df = load_metrics_json_as_dataframe(metrics_json_path)

    # basic cleanup
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    # optional local window around injection time
    if window_size is not None:
        pre_df = df[df["time"] < inject_time].tail(window_size)
        post_df = df[df["time"] >= inject_time].head(window_size)
        df = pd.concat([pre_df, post_df], ignore_index=True)

    # normalize latency naming
    df = df.loc[:, ~df.columns.str.endswith("_latency-50")]
    df = df.rename(
        columns={
            c: c.replace("_latency-90", "_latency")
            for c in df.columns
            if c.endswith("_latency-90")
        }
    )

    # RE1 / RE2 use explicit issue names in the folder
    # RE3 uses fault ids like f1/f2/f3, so metric family is not directly encoded
    issue_map = {
        "cpu": "cpu",
        "mem": "mem",
        "delay": "latency",
        "loss": "latency",
        "disk": "latency",
        "latency": "latency",
        "socket": "socket",
    }

    if fault_label in issue_map:
        metric_suffix = issue_map[fault_label]
    elif dataset_prefix.startswith("re3"):
        metric_suffix = None
    else:
        raise ValueError(
            f"Unsupported fault label '{fault_label}' in case '{case_name}'"
        )

    normal_data = df[df["time"] < inject_time].copy()
    anomaly_data = df[df["time"] >= inject_time].copy()

    if metric_suffix is not None:
        selected_cols = [c for c in df.columns if c.endswith("_" + metric_suffix)]

        if len(selected_cols) > 0:
            normal_data = normal_data.loc[:, normal_data.columns.str.endswith("_" + metric_suffix)]
            anomaly_data = anomaly_data.loc[:, anomaly_data.columns.str.endswith("_" + metric_suffix)]

            normal_data.rename(
                columns={c: c[: -(len(metric_suffix) + 1)] for c in normal_data.columns},
                inplace=True,
            )
            anomaly_data.rename(
                columns={c: c[: -(len(metric_suffix) + 1)] for c in anomaly_data.columns},
                inplace=True,
            )
        else:
            print(
                f"[Warning] No '*_{metric_suffix}' columns found for case {case_name}. "
                f"Using all metrics instead."
            )
            normal_data = normal_data.drop(columns=["time"], errors="ignore")
            anomaly_data = anomaly_data.drop(columns=["time"], errors="ignore")
    else:
        # RE3: keep all metric columns except time
        normal_data = normal_data.drop(columns=["time"], errors="ignore")
        anomaly_data = anomaly_data.drop(columns=["time"], errors="ignore")

    if "rabbitmq-exporter" in normal_data.columns:
        normal_data.drop(columns=["rabbitmq-exporter"], inplace=True)
    if "rabbitmq-exporter" in anomaly_data.columns:
        anomaly_data.drop(columns=["rabbitmq-exporter"], inplace=True)

    normal_data, anomaly_data = drop_degenerate_columns(normal_data, anomaly_data)

    if normal_data.shape[1] == 0:
        raise ValueError(
            f"No usable columns left for fault_label={fault_label} in case {case_name}"
        )

    target_node = infer_target_node(dataset_prefix, normal_data.columns)

    if target_node not in anomaly_data.columns:
        if root_cause in anomaly_data.columns:
            target_node = root_cause
        else:
            target_node = anomaly_data.columns[0]

    if not use_all_anomaly_samples:
        scorer = ITAnomalyScorer(anomaly_scorer())
        scorer.fit(normal_data[target_node].to_numpy())

        detection_index = None
        max_score_index = None
        max_score = -np.inf

        for i, value in enumerate(anomaly_data[target_node]):
            score = scorer.score(np.array([value]))
            if score > max_score:
                max_score = score
                max_score_index = i
            if value > anomaly_threshold:
                detection_index = i
                break

        if detection_index is None:
            detection_index = max_score_index

        anomaly_data = anomaly_data.iloc[[detection_index]]

    if graph_builder is None:
        graph = make_empty_graph(normal_data.columns)
    else:
        graph = prune_graph_to_observed_nodes(graph_builder(), normal_data.columns)

    import pdb; pdb.set_trace()
    return {
        "graph": graph,
        "training_sample": normal_data,
        "anomaly_sample": anomaly_data,
        "root_cause": root_cause,
        "target_node": target_node,
    }