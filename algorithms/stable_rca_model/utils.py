import numpy as np
import random
from math import ceil
import torch
from sklearn.metrics import mean_squared_error
import sys
import pandas as pd

def weighted_cov(X, W):
    '''
    X: numpy array, (n, p)
    W: numpy array, (n, 1), sum up to 1
    '''
    X_bar =  np.matmul(X.T, W) # shape: (p, 1)
    return np.matmul(X.T, W*X) - np.matmul(X_bar, X_bar.T)

def weighted_cov_torch(X, Y=None, W=None):
    if Y is None:
        X_bar = torch.matmul(X.T, W) # shape: (p, 1)
        return torch.matmul(X.T, W*X) - torch.matmul(X_bar, X_bar.T)
    else:
        X_bar = torch.matmul(X.T, W) # shape: (p, 1)
        Y_bar = torch.matmul(Y.T, W) # shape: (p, 1)
        return torch.matmul(X.T, W*Y) - torch.matmul(X_bar, Y_bar.T)
    
def get_cov_mask(select_ratio):
    select_ratio = np.reshape(select_ratio, (-1, 1))
    cov_mask = 1-np.matmul(select_ratio, select_ratio.T)
    return cov_mask 

def root_cause_detection(shift_dict,
                         pred_dict,
                         metric='mse',
                         threshold=0.2):
    shift_metric = shift_dict[f'{metric}']
    pred_ref_metric = pred_dict[f'{metric}_df_eval']

    # If the metric shift is greater than the theshold ratio multiplied 
    # by the prediction metric evaluated on observational data, the node 
    # is selected as root cause 
    # print(f"shift metric, {shift_metric}")
    # print(f"ref metric, {pred_ref_metric}")
    if shift_metric > (threshold * pred_ref_metric):
        return True
    else:
        return False
    
def detection_metrics(detected_nodes, gt_nodes):
    detected_set = set(detected_nodes)
    gt_set = set(gt_nodes)
    
    tp = len(detected_set & gt_set)
    fp = len(detected_set - gt_set)
    fn = len(gt_set - detected_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def pretty(vector):
    if type(vector) is list:
        vlist = vector
    elif type(vector) is np.ndarray:
        vlist = vector.reshape(-1).tolist()
    else:
        vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate writing
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

## Find specific nodes in a graph
def get_all_descendants(graph, node):
    visited = set()
    stack = [node]

    descendants = set()

    while stack:
        current = stack.pop()
        for child in graph.successors(current):
            if child not in visited:
                visited.add(child)
                descendants.add(child)
                stack.append(child)
    return descendants

def get_all_antecedents(graph, node):
    visited = set()
    stack = [node]

    antecedents = set()

    while stack:
        current = stack.pop()
        for parent in graph.predecessors(current):
            if parent not in visited:
                visited.add(parent)
                antecedents.add(parent)
                stack.append(parent)
    return antecedents

def get_parents(graph, node):
    return set(graph.predecessors(node))

def get_grandparents(graph, node):
    grandparents = set()
    for parent in graph.predecessors(node):
        for grandparent in graph.predecessors(parent):
            grandparents.add(grandparent)
    return grandparents

def get_siblings(graph, node):
    siblings = set()
    for parent in graph.predecessors(node):
        for child in graph.successors(parent):
            if child != node:
                siblings.add(child)
    return siblings

def get_markov_blanket(G, node):
    """Return the Markov blanket of a node in a DAG."""
    # Parents
    parents = set(G.predecessors(node))
    
    # Children
    children = set(G.successors(node))
    
    # Co-parents (parents of the children, excluding the node itself)
    coparents = set()
    for child in children:
        coparents.update(G.predecessors(child))
    coparents.discard(node)
    
    # Markov blanket is union of all three
    markov_blanket = parents.union(children).union(coparents)
    
    return list(markov_blanket)

def get_non_markov_blanket(G, node):
    """Return all nodes that are NOT in the Markov blanket of `node` in a DAG."""
    # Get the Markov blanket
    mb = set(get_markov_blanket(G, node))
    
    # All nodes in the graph
    all_nodes = set(G.nodes)
    
    # Remove the node itself and its MB
    non_mb_nodes = all_nodes - mb - {node}
    
    return list(non_mb_nodes)

def select_features_by_quantile(
    feature_weights_df: pd.DataFrame,
    quantile=0.9,
    use_abs=True,
    min_features=1
):
    """
    Select features whose importance is above a given quantile.

    Parameters
    ----------
    feature_weights_df : pd.DataFrame
        DataFrame with columns ["feature", "weight"].
    feature_names : np.ndarray, shape (d,)
        Feature names.
    quantile : float in (0, 1)
        Quantile threshold (e.g., 0.9 selects top 10% features).
    use_abs : bool
        Whether to use absolute feature weights.
    min_features : int
        Minimum number of features to select.

    Returns
    -------
    selected_feature_names : np.ndarray
        Names of selected features.
    threshold : float
        Quantile threshold value.
    """

    if use_abs:
        scores = feature_weights_df["weight"].abs()
    else:
        scores = feature_weights_df["weight"]

    threshold = np.quantile(scores, quantile)

    # select features above threshold
    selected_features_df = feature_weights_df[scores >= threshold].copy()

    # safety: ensure at least min_features are selected
    if selected_features_df.shape[0] < min_features:
        selected_features_df = feature_weights_df.nlargest(min_features, "weight" if not use_abs else scores.name)

    # sort descending
    selected_features_df = selected_features_df.sort_values(by="weight", ascending=False).reset_index(drop=True)

    return selected_features_df, threshold