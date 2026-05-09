#!/usr/bin/env python3
"""
Process RCA benchmark result files saved as one file per method.

Expected result files may look like:
  results/prorca/results_prorca_stable_rca.npy
  results/sockshop/results_sockshop_rcg_0.npy
  results/causalchamber/results_causalchamber_score_ordering.npy
  results/causalman/results_causalchamber_score_ordering.npy
  results/rcaeval/results_causalchamber_score_ordering.npy
  results/rcaeval/rcaeval_stable_rca.npy

Important:
  Dataset inference prioritizes directory names over file names. Therefore,
  even if a filename contains a stale dataset name, e.g.
      results/causalman/results_causalchamber_score_ordering.npy
  the dataset is inferred as causal_man because the parent directory is causalman.

Typical usage:
  python process_results.py --dir results/prorca --pattern "*.npy" --show-std
  python process_results.py --dir results --recursive --pattern "*.npy" --show-std
  python process_results.py --inputs results/rcaeval/*.npy --show-std
  python process_results.py --dir results --recursive --out-dir results/processed

Supported result layouts:
  {
    "config": {...},
    "synthetic_data": {
      "stable_rca_precision": [...],
      "stable_rca_recall": [...],
      "stable_rca_f1": [...],
      "stable_rca_ndcg": [...],
      "stable_rca_time": [...]
    }
  }

  {
    "config": {...},
    "all_cases": {
      "case_results": [
        {
          "case_name": "...",
          "status": "ok",
          "stable_rca_precision": 1.0,
          ...
        }
      ],
      "stable_rca_precision": [...],
      ...
    }
  }
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

try:
    import numpy as np
except Exception:
    np = None


MAIN_METRICS = ["time", "precision", "recall", "f1", "ndcg"]

METHOD_NAME_ALIASES = {
    "counterfactual": "counterfactual_contribution",
    "counterfactual_contribution": "counterfactual_contribution",
    "stable": "stable_rca",
}

PREFERRED_METHOD_ORDER = [
    "score_ordering",
    "smooth_traversal",
    "traversal",
    "cholesky",
    "circa",
    "counterfactual_contribution",
    "rcd",
    "epsilon_diagnosis",
    "baro",
    "rcg_0",
    "stable_rca",
]

DATASET_NAME_ALIASES = {
    "synthetic": "synthetic_data",
    "synthetic_data": "synthetic_data",
    "prorca": "pro_rca",
    "pro_rca": "pro_rca",
    "sockshop": "sock_shop",
    "sock_shop": "sock_shop",
    "causalchamber": "causal_chamber",
    "causal_chamber": "causal_chamber",
    "causalman": "causal_man",
    "causal_man": "causal_man",
    "rcaeval": "rcaeval",
}


# =============================================================================
# Loading
# =============================================================================

def load_result(path: str | Path) -> dict:
    """Load a result dictionary from npy, pickle, json, or repr-like text."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        if np is None:
            raise ImportError("numpy is required to load .npy files.")
        obj = np.load(path, allow_pickle=True)
        if hasattr(obj, "item"):
            obj = obj.item()
        if not isinstance(obj, dict):
            raise TypeError(f"Expected dict in {path}, got {type(obj)}.")
        return obj

    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as file:
            obj = pickle.load(file)
        if not isinstance(obj, dict):
            raise TypeError(f"Expected dict in {path}, got {type(obj)}.")
        return obj

    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        obj = json.loads(text)
    else:
        obj = ast.literal_eval(text)

    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}.")

    return obj


def collect_input_paths(args: argparse.Namespace) -> list[Path]:
    """Collect input files from --inputs and/or --dir/--pattern."""
    paths: list[Path] = []

    if args.inputs:
        for item in args.inputs:
            expanded = glob.glob(item, recursive=True)
            if expanded:
                paths.extend(Path(p) for p in expanded)
            else:
                paths.append(Path(item))

    if args.dir:
        root = Path(args.dir)
        if args.recursive:
            paths.extend(root.rglob(args.pattern))
        else:
            paths.extend(root.glob(args.pattern))

    valid_suffixes = {
        ".npy",
        ".pkl",
        ".pickle",
        ".json",
        ".txt",
        ".py",
        ".repr",
    }

    filtered: list[Path] = []
    seen = set()

    for path in paths:
        path = path.resolve()

        if path in seen:
            continue
        seen.add(path)

        if not path.is_file():
            continue

        if path.suffix.lower() not in valid_suffixes:
            continue

        if args.ignore_average_files and "average_results" in path.name:
            continue

        filtered.append(path)

    return sorted(filtered)


# =============================================================================
# Basic utilities
# =============================================================================

def is_number(value: Any) -> bool:
    """Return True for finite scalar numeric values."""
    if isinstance(value, bool):
        return False

    numeric_types = (int, float)

    if np is not None:
        numeric_types = numeric_types + (np.integer, np.floating)

    return isinstance(value, numeric_types) and math.isfinite(float(value))


def to_numeric_list(value: Any) -> list[float]:
    """Convert scalar/list/tuple/ndarray metric values into a list of floats."""
    if is_number(value):
        return [float(value)]

    if np is not None and isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        return [float(v) for v in value if is_number(v)]

    return []


def mean_std(values: Sequence[float]) -> tuple[Optional[float], Optional[float], int]:
    """Return population mean/std and count."""
    values = list(values)

    if not values:
        return None, None, 0

    if len(values) == 1:
        return mean(values), 0.0, 1

    return mean(values), pstdev(values), len(values)


def fmt_value(
    mu: Optional[float],
    sd: Optional[float],
    show_std: bool,
    digits: int,
) -> str:
    """Format mean or mean±std."""
    if mu is None:
        return "-"

    if show_std:
        return f"{mu:.{digits}f}±{sd:.{digits}f}"

    return f"{mu:.{digits}f}"


def normalize_method_name(method: str) -> str:
    """Normalize method aliases to result-key prefixes."""
    method = str(method).strip().replace("-", "_")
    return METHOD_NAME_ALIASES.get(method, method)


def normalize_dataset_name(dataset: str) -> str:
    """Normalize dataset aliases."""
    dataset = str(dataset).strip().replace("-", "_")
    return DATASET_NAME_ALIASES.get(dataset, dataset)


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    output = []

    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


def sort_methods(methods: Iterable[str]) -> list[str]:
    """Sort methods according to preferred table order."""
    order = {method: idx for idx, method in enumerate(PREFERRED_METHOD_ORDER)}
    return sorted(methods, key=lambda method: (order.get(method, 10_000), method))


def case_label(case: Any) -> str:
    """Convert arbitrary case key to a printable label."""
    if isinstance(case, tuple):
        return " | ".join(map(str, case))
    return str(case)


def get_graph_mode(config: Mapping[str, Any]) -> str:
    """
    Return graph_mode.

    This intentionally uses only graph_mode. Deprecated flags such as
    use_xges_dag are ignored.
    """
    return str(config.get("graph_mode", "unknown"))


# =============================================================================
# Method and dataset inference
# =============================================================================

def methods_from_config(config: Mapping[str, Any]) -> list[str]:
    """Infer method names from result['config']."""
    raw = config.get("methods", None)

    if raw is None:
        raw = config.get("method", None)

    if isinstance(raw, str):
        return [
            normalize_method_name(item)
            for item in raw.split(",")
            if item.strip()
        ]

    if isinstance(raw, (list, tuple, set)):
        return [
            normalize_method_name(str(item))
            for item in raw
            if str(item).strip()
        ]

    return []


def is_stable_internal_prefix(prefix: str) -> bool:
    """
    Return True if a metric prefix is an internal StableRCA diagnostic rather
    than the method name stable_rca.
    """
    return prefix.startswith("stable_rca_") and prefix != "stable_rca"


def methods_from_metric_keys(metrics: Mapping[str, Any]) -> list[str]:
    """Infer method names from flat or nested metric keys."""
    candidates = set()

    for key in metrics.keys():
        if not isinstance(key, str):
            continue

        for metric in MAIN_METRICS:
            suffix = f"_{metric}"

            if key.endswith(suffix):
                prefix = key[: -len(suffix)]

                if not prefix:
                    continue

                if is_stable_internal_prefix(prefix):
                    continue

                candidates.add(normalize_method_name(prefix))

    for key, value in metrics.items():
        if isinstance(key, str) and isinstance(value, Mapping):
            if any(metric in value for metric in MAIN_METRICS):
                candidates.add(normalize_method_name(key))

    return sort_methods(candidates)


def method_from_filename(path: Path) -> Optional[str]:
    """Infer method name from filename."""
    stem = path.stem.replace("-", "_")

    known_methods = sorted(
        set(PREFERRED_METHOD_ORDER) | set(METHOD_NAME_ALIASES.keys()),
        key=len,
        reverse=True,
    )

    for method in known_methods:
        pattern = rf"(^|[_\-.]){re.escape(method)}($|[_\-.])"
        if re.search(pattern, stem):
            return normalize_method_name(method)

    return None


def dataset_from_filename(path: Path) -> Optional[str]:
    """
    Infer dataset name from path.

    Directory names are prioritized over filename because some result files may
    have stale names, e.g.

        results/causalman/results_causalchamber_score_ordering.npy
        results/rcaeval/results_causalchamber_score_ordering.npy

    In these cases, the parent directory is more reliable than the filename.
    """
    known_datasets = sorted(
        DATASET_NAME_ALIASES.keys(),
        key=len,
        reverse=True,
    )

    # Prefer directory names, nearest parent first.
    parent_parts = [
        part.replace("-", "_")
        for part in path.parent.parts[::-1]
    ]

    for part in parent_parts:
        for dataset in known_datasets:
            if part == dataset:
                return normalize_dataset_name(dataset)

    # Fallback: search the full parent path.
    parent_text = "_".join(parent_parts)

    for dataset in known_datasets:
        pattern = rf"(^|[_\-.\/]){re.escape(dataset)}($|[_\-.\/])"
        if re.search(pattern, parent_text):
            return normalize_dataset_name(dataset)

    # Last fallback: use filename stem.
    stem = path.stem.replace("-", "_")

    for dataset in known_datasets:
        pattern = rf"(^|[_\-.]){re.escape(dataset)}($|[_\-.])"
        if re.search(pattern, stem):
            return normalize_dataset_name(dataset)

    return None


def infer_dataset(result: Mapping[Any, Any], path: Path) -> str:
    """
    Infer dataset/experiment mode.

    The directory name is preferred over config and filename, because some files
    may have stale dataset names in the filename or copied config.
    """
    inferred_from_path = dataset_from_filename(path)

    if inferred_from_path is not None:
        return inferred_from_path

    config = result.get("config", {})

    if isinstance(config, Mapping):
        for key in ["experiment_mode", "dataset", "dataset_name"]:
            if config.get(key) is not None:
                return normalize_dataset_name(str(config[key]))

    return "unknown"


def infer_methods_for_result(
    result: Mapping[Any, Any],
    path: Path,
) -> list[str]:
    """Infer methods from config, metric keys, case_results, or filename."""
    methods: list[str] = []

    config = result.get("config", {})
    if isinstance(config, Mapping):
        methods.extend(methods_from_config(config))

    for case, metrics in result.items():
        if case == "config":
            continue

        if isinstance(metrics, Mapping):
            methods.extend(methods_from_metric_keys(metrics))

            case_results = metrics.get("case_results")
            if isinstance(case_results, list):
                for row in case_results[:20]:
                    if isinstance(row, Mapping):
                        methods.extend(methods_from_metric_keys(row))

    if not methods:
        method = method_from_filename(path)
        if method is not None:
            methods.append(method)

    return sort_methods(unique_preserve_order(methods))


# =============================================================================
# Record extraction
# =============================================================================

Record = dict[str, Any]


def extract_case_id_from_row(row: Mapping[str, Any], fallback_idx: int) -> str:
    """Build a case id from an RCAEval-style row."""
    for key in ["case_name", "case", "scenario", "dataset", "name"]:
        if row.get(key) is not None:
            return str(row[key])

    pieces = []

    for key in ["dataset_prefix", "root_cause", "fault_label", "instance_id", "trial"]:
        if row.get(key) is not None:
            pieces.append(str(row[key]))

    if pieces:
        return "_".join(pieces)

    return f"case_{fallback_idx}"


def get_metric_values_from_mapping(
    metrics: Mapping[str, Any],
    method: str,
    metric: str,
) -> list[float]:
    """Extract values for one method/metric from flat or nested layout."""
    values: list[float] = []

    values.extend(to_numeric_list(metrics.get(f"{method}_{metric}")))

    nested = metrics.get(method)
    if isinstance(nested, Mapping):
        values.extend(to_numeric_list(nested.get(metric)))

    return values


def extract_records_from_case_results(
    result_entry: Mapping[str, Any],
    *,
    dataset: str,
    graph_mode: str,
    source_file: Path,
    methods: Sequence[str],
    include_failed_as_zero: bool,
) -> list[Record]:
    """Extract records from result_entry['case_results']."""
    rows = result_entry.get("case_results")

    if not isinstance(rows, list):
        return []

    records: list[Record] = []

    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue

        status = str(row.get("status", "ok"))

        if status != "ok" and not include_failed_as_zero:
            continue

        case = extract_case_id_from_row(row, idx)

        for method in methods:
            values_by_metric: dict[str, list[float]] = {}
            has_any_value = False

            for metric in MAIN_METRICS:
                values = get_metric_values_from_mapping(row, method, metric)

                if status != "ok" and include_failed_as_zero and metric != "time":
                    values = [0.0]

                values_by_metric[metric] = values
                has_any_value = has_any_value or bool(values)

            if has_any_value:
                records.append(
                    {
                        "dataset": dataset,
                        "case": case,
                        "method": method,
                        "graph_mode": graph_mode,
                        "values": values_by_metric,
                        "source_file": str(source_file),
                    }
                )

    return records


def extract_records_from_metric_entry(
    case: Any,
    metrics: Mapping[str, Any],
    *,
    dataset: str,
    graph_mode: str,
    source_file: Path,
    methods: Sequence[str],
) -> list[Record]:
    """Extract records from a normal scenario/case metric dictionary."""
    records: list[Record] = []

    for method in methods:
        values_by_metric: dict[str, list[float]] = {}
        has_any_value = False

        for metric in MAIN_METRICS:
            values = get_metric_values_from_mapping(metrics, method, metric)
            values_by_metric[metric] = values
            has_any_value = has_any_value or bool(values)

        if has_any_value:
            records.append(
                {
                    "dataset": dataset,
                    "case": case_label(case),
                    "method": method,
                    "graph_mode": graph_mode,
                    "values": values_by_metric,
                    "source_file": str(source_file),
                }
            )

    return records


def extract_records(
    result: Mapping[Any, Any],
    path: Path,
    include_failed_as_zero: bool = False,
) -> tuple[list[Record], list[str]]:
    """Extract all metric records from one result dictionary."""
    config = result.get("config", {})
    if not isinstance(config, Mapping):
        config = {}

    dataset = infer_dataset(result, path)
    graph_mode = get_graph_mode(config)
    methods = infer_methods_for_result(result, path)

    records: list[Record] = []

    for case, metrics in result.items():
        if case == "config":
            continue

        if not isinstance(metrics, Mapping):
            continue

        case_result_records = extract_records_from_case_results(
            metrics,
            dataset=dataset,
            graph_mode=graph_mode,
            source_file=path,
            methods=methods,
            include_failed_as_zero=include_failed_as_zero,
        )

        if case_result_records:
            records.extend(case_result_records)
            continue

        records.extend(
            extract_records_from_metric_entry(
                case,
                metrics,
                dataset=dataset,
                graph_mode=graph_mode,
                source_file=path,
                methods=methods,
            )
        )

    return records, methods


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_records_by_dataset_case_method(
    records: Sequence[Record],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Aggregate raw records by dataset, case, and method."""
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}

    for record in records:
        key = (
            str(record["dataset"]),
            str(record["case"]),
            str(record["method"]),
        )

        if key not in grouped:
            grouped[key] = {
                "dataset": str(record["dataset"]),
                "case": str(record["case"]),
                "method": str(record["method"]),
                "graph_mode": str(record.get("graph_mode", "unknown")),
                "values": {metric: [] for metric in MAIN_METRICS},
                "source_files": set(),
            }

        grouped[key]["source_files"].add(record.get("source_file", ""))

        old_graph_mode = grouped[key]["graph_mode"]
        new_graph_mode = str(record.get("graph_mode", "unknown"))

        if old_graph_mode != new_graph_mode:
            grouped[key]["graph_mode"] = "mixed"

        for metric in MAIN_METRICS:
            grouped[key]["values"][metric].extend(
                record.get("values", {}).get(metric, [])
            )

    return grouped


def compute_per_case_table(
    grouped: Mapping[tuple[str, str, str], Mapping[str, Any]],
    methods: Sequence[str],
    show_std: bool,
    digits: int,
) -> list[dict[str, str]]:
    """Compute per-case averages."""
    datasets = sorted({dataset for dataset, _, _ in grouped.keys()})
    rows: list[dict[str, str]] = []

    for dataset in datasets:
        cases = sorted({case for d, case, _ in grouped.keys() if d == dataset})

        for case in cases:
            for method in methods:
                item = grouped.get((dataset, case, method))
                if item is None:
                    continue

                row: dict[str, str] = {
                    "dataset": dataset,
                    "case": case,
                    "method": method,
                    "graph_mode": str(item.get("graph_mode", "unknown")),
                }

                has_any = False
                values = item.get("values", {})

                for metric in MAIN_METRICS:
                    mu, sd, n = mean_std(values.get(metric, []))
                    row[metric] = fmt_value(mu, sd, show_std, digits)
                    row[f"n_{metric}"] = str(n)
                    has_any = has_any or n > 0

                if has_any:
                    rows.append(row)

    return rows


def compute_dataset_overall_table(
    grouped: Mapping[tuple[str, str, str], Mapping[str, Any]],
    methods: Sequence[str],
    show_std: bool,
    digits: int,
    macro: bool,
) -> list[dict[str, str]]:
    """Compute averages by dataset and method."""
    datasets = sorted({dataset for dataset, _, _ in grouped.keys()})
    rows: list[dict[str, str]] = []

    for dataset in datasets:
        for method in methods:
            items = [
                item
                for (d, _, m), item in grouped.items()
                if d == dataset and m == method
            ]

            if not items:
                continue

            graph_modes = sorted(
                {str(item.get("graph_mode", "unknown")) for item in items}
            )
            graph_mode = graph_modes[0] if len(graph_modes) == 1 else "mixed"

            row: dict[str, str] = {
                "dataset": dataset,
                "method": method,
                "graph_mode": graph_mode,
            }

            for metric in MAIN_METRICS:
                if macro:
                    values_to_average = []
                    for item in items:
                        mu, _, _ = mean_std(item.get("values", {}).get(metric, []))
                        if mu is not None:
                            values_to_average.append(mu)
                else:
                    values_to_average = []
                    for item in items:
                        values_to_average.extend(
                            item.get("values", {}).get(metric, [])
                        )

                mu, sd, n = mean_std(values_to_average)
                row[metric] = fmt_value(mu, sd, show_std, digits)
                row[f"n_{metric}"] = str(n)

            rows.append(row)

    return rows


def compute_global_overall_table(
    grouped: Mapping[tuple[str, str, str], Mapping[str, Any]],
    methods: Sequence[str],
    show_std: bool,
    digits: int,
    macro: bool,
) -> list[dict[str, str]]:
    """Compute averages by method across all loaded datasets."""
    rows: list[dict[str, str]] = []

    for method in methods:
        items = [
            item
            for (_, _, m), item in grouped.items()
            if m == method
        ]

        if not items:
            continue

        graph_modes = sorted(
            {str(item.get("graph_mode", "unknown")) for item in items}
        )
        graph_mode = graph_modes[0] if len(graph_modes) == 1 else "mixed"

        row: dict[str, str] = {
            "method": method,
            "graph_mode": graph_mode,
        }

        for metric in MAIN_METRICS:
            if macro:
                values_to_average = []
                for item in items:
                    mu, _, _ = mean_std(item.get("values", {}).get(metric, []))
                    if mu is not None:
                        values_to_average.append(mu)
            else:
                values_to_average = []
                for item in items:
                    values_to_average.extend(item.get("values", {}).get(metric, []))

            mu, sd, n = mean_std(values_to_average)
            row[metric] = fmt_value(mu, sd, show_std, digits)
            row[f"n_{metric}"] = str(n)

        rows.append(row)

    return rows


def compute_stable_rca_details(
    result_by_path: Mapping[Path, Mapping[Any, Any]],
    show_std: bool,
    digits: int,
) -> list[dict[str, str]]:
    """Average StableRCA diagnostic metrics."""
    exclude = {f"stable_rca_{metric}" for metric in MAIN_METRICS}
    values_by_dataset_metric: MutableMapping[tuple[str, str], list[float]] = defaultdict(list)

    for path, result in result_by_path.items():
        dataset = infer_dataset(result, path)

        for case, metrics in result.items():
            if case == "config" or not isinstance(metrics, Mapping):
                continue

            for key, value in metrics.items():
                if (
                    isinstance(key, str)
                    and key.startswith("stable_rca_")
                    and key not in exclude
                ):
                    values_by_dataset_metric[(dataset, key)].extend(
                        to_numeric_list(value)
                    )

            case_results = metrics.get("case_results")
            if isinstance(case_results, list):
                for row in case_results:
                    if not isinstance(row, Mapping):
                        continue
                    if row.get("status", "ok") != "ok":
                        continue
                    for key, value in row.items():
                        if (
                            isinstance(key, str)
                            and key.startswith("stable_rca_")
                            and key not in exclude
                        ):
                            values_by_dataset_metric[(dataset, key)].extend(
                                to_numeric_list(value)
                            )

    rows = []

    for dataset, metric in sorted(values_by_dataset_metric):
        values = values_by_dataset_metric[(dataset, metric)]
        mu, sd, n = mean_std(values)

        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "mean": fmt_value(mu, sd, show_std, digits),
                "n": str(n),
            }
        )

    return rows


def compute_file_summary(
    file_methods: Mapping[Path, Sequence[str]],
    file_record_counts: Mapping[Path, int],
) -> list[dict[str, str]]:
    """Summarize loaded files."""
    rows = []

    for path in sorted(file_methods):
        rows.append(
            {
                "file": str(path),
                "methods": ",".join(file_methods[path]),
                "records": str(file_record_counts.get(path, 0)),
            }
        )

    return rows


# =============================================================================
# Output
# =============================================================================

def print_table(
    rows: Sequence[Mapping[str, Any]],
    title: str,
    columns: Sequence[str],
) -> None:
    """Pretty-print a table."""
    print("\n" + title)
    print("=" * len(title))

    if not rows:
        print("[empty]")
        return

    widths = {col: len(col) for col in columns}

    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    header = "  ".join(col.ljust(widths[col]) for col in columns)
    sep = "  ".join("-" * widths[col] for col in columns)

    print(header)
    print(sep)

    for row in rows:
        print("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))


def write_csv(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    columns: Sequence[str],
) -> None:
    """Write rows to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(columns))
        writer.writeheader()

        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_markdown(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    columns: Sequence[str],
    title: str,
) -> None:
    """Write rows to Markdown."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"# {title}", ""]

    if not rows:
        lines.append("[empty]")
    else:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

        for row in rows:
            lines.append(
                "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process RCA result files saved separately by method."
    )

    parser.add_argument(
        "--dir",
        type=str,
        default="results/synthetic_data",
        help="Directory containing result files.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.npy",
        help=(
            "Glob pattern inside --dir. Examples: '*.npy', 'results_*_*.npy', "
            "'rcaeval_*.npy'."
        ),
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search --dir using --pattern.",
    )

    parser.add_argument(
        "--inputs",
        type=str,
        nargs="*",
        default=None,
        help="Explicit input files or glob patterns. Can be used with --dir.",
    )

    parser.add_argument(
        "--ignore-average-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore files whose names contain 'average_results'.",
    )

    parser.add_argument(
        "--show-std",
        action="store_true",
        help="Print mean±std instead of mean only.",
    )

    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of decimal places.",
    )

    parser.add_argument(
        "--macro",
        action="store_true",
        help="Average per-case means instead of concatenating all trial values.",
    )

    parser.add_argument(
        "--include-failed-as-zero",
        action="store_true",
        help="Count failed RCAEval-style case_results as zero for ranking metrics.",
    )

    parser.add_argument(
        "--stable-details",
        action="store_true",
        help="Also print averaged StableRCA diagnostic metrics.",
    )

    parser.add_argument(
        "--print-file-summary",
        action="store_true",
        help="Print loaded files, inferred methods, and extracted record counts.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional directory to save CSV and Markdown tables.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = collect_input_paths(args)

    if not paths:
        raise FileNotFoundError(
            f"No result files found. Check --dir={args.dir!r}, "
            f"--pattern={args.pattern!r}, --recursive, or --inputs."
        )

    result_by_path: dict[Path, Mapping[Any, Any]] = {}
    all_records: list[Record] = []
    file_methods: dict[Path, list[str]] = {}
    file_record_counts: dict[Path, int] = {}

    for path in paths:
        result = load_result(path)
        result_by_path[path] = result

        records, methods = extract_records(
            result,
            path,
            include_failed_as_zero=args.include_failed_as_zero,
        )

        file_methods[path] = methods
        file_record_counts[path] = len(records)
        all_records.extend(records)

    if not all_records:
        raise ValueError(
            "Loaded result files, but could not extract any metric records."
        )

    grouped = aggregate_records_by_dataset_case_method(all_records)
    methods = sort_methods(
        unique_preserve_order(record["method"] for record in all_records)
    )

    global_columns = [
        "method",
        "graph_mode",
        "time",
        "precision",
        "recall",
        "f1",
        "ndcg",
    ]

    dataset_columns = [
        "dataset",
        "method",
        "graph_mode",
        "time",
        "precision",
        "recall",
        "f1",
        "ndcg",
    ]

    per_case_columns = [
        "dataset",
        "case",
        "method",
        "graph_mode",
        "time",
        "precision",
        "recall",
        "f1",
        "ndcg",
    ]

    file_summary_columns = ["file", "methods", "records"]

    global_rows = compute_global_overall_table(
        grouped,
        methods,
        show_std=args.show_std,
        digits=args.digits,
        macro=args.macro,
    )

    dataset_rows = compute_dataset_overall_table(
        grouped,
        methods,
        show_std=args.show_std,
        digits=args.digits,
        macro=args.macro,
    )

    per_case_rows = compute_per_case_table(
        grouped,
        methods,
        show_std=args.show_std,
        digits=args.digits,
    )

    file_summary_rows = compute_file_summary(file_methods, file_record_counts)

    if args.print_file_summary:
        print_table(file_summary_rows, "Loaded result files", file_summary_columns)

    print_table(global_rows, "Global average results", global_columns)
    print_table(dataset_rows, "Dataset average results", dataset_columns)
    print_table(per_case_rows, "Per-case average results", per_case_columns)

    stable_detail_rows: list[dict[str, str]] = []

    if args.stable_details:
        stable_detail_rows = compute_stable_rca_details(
            result_by_path,
            show_std=args.show_std,
            digits=args.digits,
        )
        print_table(
            stable_detail_rows,
            "StableRCA diagnostic averages",
            ["dataset", "metric", "mean", "n"],
        )

    print("\nSummary")
    print("=======")
    print(f"Loaded files: {len(paths)}")
    print(f"Extracted raw records: {len(all_records)}")
    print(f"Aggregated dataset-case-method pairs: {len(grouped)}")
    print(f"Methods: {', '.join(methods)}")

    if args.out_dir:
        out_dir = Path(args.out_dir)

        write_csv(global_rows, out_dir / "global_average_results.csv", global_columns)
        write_csv(dataset_rows, out_dir / "dataset_average_results.csv", dataset_columns)
        write_csv(per_case_rows, out_dir / "per_case_average_results.csv", per_case_columns)
        write_csv(file_summary_rows, out_dir / "loaded_result_files.csv", file_summary_columns)

        write_markdown(
            global_rows,
            out_dir / "global_average_results.md",
            global_columns,
            "Global average results",
        )

        write_markdown(
            dataset_rows,
            out_dir / "dataset_average_results.md",
            dataset_columns,
            "Dataset average results",
        )

        write_markdown(
            per_case_rows,
            out_dir / "per_case_average_results.md",
            per_case_columns,
            "Per-case average results",
        )

        write_markdown(
            file_summary_rows,
            out_dir / "loaded_result_files.md",
            file_summary_columns,
            "Loaded result files",
        )

        if stable_detail_rows:
            write_csv(
                stable_detail_rows,
                out_dir / "stable_rca_diagnostic_averages.csv",
                ["dataset", "metric", "mean", "n"],
            )
            write_markdown(
                stable_detail_rows,
                out_dir / "stable_rca_diagnostic_averages.md",
                ["dataset", "metric", "mean", "n"],
                "StableRCA diagnostic averages",
            )

        print(f"\nSaved tables to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()