# ./result_saver.py

import os
import numpy as np
from warnings import warn


def get_unique_filename(path):
    directory, base = os.path.split(path)
    name, ext = os.path.splitext(base)
    if not ext:
        ext = '.npy'  # Default extension if none is given
    i = 0
    while True:
        filename = f"{name}_{i}{ext}" if i > 0 else f"{name}{ext}"
        full_path = os.path.join(directory, filename)
        if not os.path.exists(full_path):
            return full_path
        i += 1


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(get_unique_filename(path), results)


def load_results(path):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
    warn(f"{path} not found, returning an empty dictionary.")
    return {}

def compute_average_results(results):
    """
    Compute the average of each numeric metric for each method in the results dictionary.

    Args:
        results (dict): Dictionary containing 'config' and results keyed by parameters.

    Returns:
        dict: Nested dictionary with structure {param_key: {method: {metric: average_value}}}
    """
    import numpy as np

    average_results = {}

    for param_key, result in results.items():
        if param_key == "config":
            continue

        average_results[param_key] = {}

        for key, values in result.items():
            # Skip metadata / per-case records
            if key in {"case_results", "case_metadata"}:
                continue

            # Only process non-empty numeric lists
            if not isinstance(values, list) or len(values) == 0:
                continue

            if not all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
                continue

            method_name, metric_name = key.rsplit("_", 1)

            if method_name not in average_results[param_key]:
                average_results[param_key][method_name] = {}

            average_results[param_key][method_name][metric_name] = float(np.mean(values))

    return average_results
