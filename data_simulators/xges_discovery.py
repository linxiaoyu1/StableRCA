import numpy as np
import networkx as nx
import pandas as pd

def drop_low_variance(df, eps=1e-8):
    """
    Remove columns with (near-)zero variance.

    Returns
    -------
    df_reduced : pd.DataFrame
    keep_mask : np.ndarray[bool]
    """
    var = df.var()
    keep_mask = (var > eps).to_numpy()
    return df.loc[:, keep_mask], keep_mask

def drop_collinear_columns(X, tol=1e-10):
    """
    Remove linearly dependent columns using rank-revealing QR.

    Returns
    -------
    X_reduced : np.ndarray
    keep_mask : np.ndarray[bool]
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")

    Q, R = np.linalg.qr(X)
    keep_mask = np.abs(np.diag(R)) > tol

    # Safety: ensure at least one column survives
    if not np.any(keep_mask):
        raise ValueError("All columns removed due to collinearity")

    return X[:, keep_mask], keep_mask

def add_edges_by_correlation(G, df, target, tau=0.2):
    y = df[target].values

    for n in G.nodes:
        if n == target:
            continue
        x = df[n].values

        if np.std(x) < 1e-8:
            continue

        corr = np.corrcoef(x, y)[0, 1]
        if np.isfinite(corr) and abs(corr) > tau:
            G.add_edge(n, target)

    return G

def df_to_xges_digraph(df: pd.DataFrame, label_col = None, target = None) -> nx.DiGraph:
    from xges import XGES
    if target is not None and target not in df.columns:
        raise ValueError(f"target '{target}' not in DataFrame")
    
    if label_col:
        feat_cols = [c for c in df.columns if c != label_col]
    else:
        feat_cols = (list)(df.columns)
    
    df_feat = df[feat_cols].copy()
    # Track whether target is present
    target_initially_present = target in df_feat.columns if target else False

    # 2. Drop low-variance columns (keep mask!)
    df_feat, keep_lv = drop_low_variance(df_feat)
    target_dropped_lv = target_initially_present and target not in df_feat.columns
    feat_cols = list(df_feat.columns)

    # 3. Drop collinear columns (keep mask!)
    X = df_feat.to_numpy(dtype=np.float64)
    X, keep_col = drop_collinear_columns(X)
    feat_cols = [c for c, k in zip(feat_cols, keep_col) if k]
    target_dropped_col = target_initially_present and target not in feat_cols
    print(feat_cols)

    # Final decision
    target_dropped = target_dropped_lv or target_dropped_col

    if X.ndim != 2:
        raise ValueError(f"Expect 2D feature matrix, got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN/inf. Please impute/clean first.")
    
    xges = XGES()
    pdag = xges.fit(X)
    dag = pdag.get_dag_extension()
    G = dag.to_networkx()

    mapping = {i: feat_cols[i] for i in range(len(feat_cols))}
    G = nx.relabel_nodes(G, mapping, copy=True)

    # add back all dropped nodes
    all_vars = list(df.columns)

    for v in all_vars:
        if v not in G:
            G.add_node(v)
    
    # Re-add target as sink if it was dropped
    if target and target_dropped:
        G = add_edges_by_correlation(G, df, target)
        
    return G