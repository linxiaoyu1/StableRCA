from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .dwr import DWR
from .propensity_score import calculate_propensity_weights
from .srdo import SRDO
from .STG import STG
from .utils import get_cov_mask


TaskType = Literal["regression", "classification"]
PredictionModelType = Literal["stg", "catboost", "linear"]

EPS = 1e-12
FEATURE_SELECTION_RANDOM_STATE = 81
CDS_RANDOM_STATE = 42


# =============================================================================
# Helper functions
# =============================================================================

def _get_stable_weights(
    X_train_scaled: np.ndarray,
    args: Any,
) -> np.ndarray:
    """Compute sample weights using the selected stable weighting algorithm."""
    algorithm = args.stable_weighting_algorithm

    if algorithm == "dwr":
        weights = DWR(
            X_train_scaled,
            order=args.order,
            num_steps=args.num_steps,
        )
    elif algorithm == "srdo":
        weights = SRDO(X_train_scaled, p_s=1)
    else:
        raise ValueError(f"Unknown stable_weighting_algorithm: {algorithm}")

    return np.asarray(weights).reshape(-1)


def _select_features_by_weight_ratio(
    feature_names: np.ndarray,
    feature_weights: np.ndarray,
    args: Any,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Select features whose absolute weight is larger than
    max_abs_weight * args.selection_weight_ratio.

    If all weights are zero, or no feature passes the threshold, this function
    keeps the top feature to avoid returning an empty conditioning set unless
    there are no input features.
    """
    feature_weights = np.asarray(feature_weights, dtype=float).reshape(-1)

    feature_weights_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "weight": feature_weights,
            }
        )
        .sort_values(by="weight", ascending=False)
        .reset_index(drop=True)
    )

    if len(feature_weights) == 0:
        return np.array([], dtype=object), feature_weights_df

    abs_weights = np.abs(feature_weights)
    max_abs_weight = np.max(abs_weights)

    if max_abs_weight <= EPS:
        selected_indices = np.array([0])
    else:
        threshold = max_abs_weight * args.selection_weight_ratio
        selected_indices = np.where(abs_weights > threshold)[0]

        if len(selected_indices) == 0:
            selected_indices = np.array([int(np.argmax(abs_weights))])

    selected_feature_names = feature_names[selected_indices]
    return selected_feature_names, feature_weights_df


def _fit_stg_feature_weights(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    args: Any,
) -> np.ndarray:
    """
    Iterative STG feature selection with DWR-based covariance masking.

    This follows the original implementation:
    - initialize covariance mask with all-zero feature ratios;
    - iteratively run DWR and STG;
    - smooth feature selection ratios using a moving average.
    """
    num_features = X_train_scaled.shape[1]
    num_epochs = args.num_epoch

    if num_epochs <= 0:
        raise ValueError("args.num_epoch must be positive when using STG.")

    cov_mask = get_cov_mask(np.zeros(num_features))
    select_ratio_ma = []

    for _ in range(num_epochs):
        weights = DWR(
            X_train_scaled,
            cov_mask=cov_mask,
            order=args.order,
            num_steps=args.num_steps,
        )

        stg = STG(
            num_features,
            1,
            sigma=args.sigma_STG,
            lam=args.lam_STG,
        )

        stg.train(
            X_train_scaled,
            y_train,
            W=weights,
            epochs=getattr(args, "stg_epochs", 5000),
        )

        select_ratio = stg.get_ratios().detach().cpu().numpy()

        if len(select_ratio_ma) >= args.period_MA:
            select_ratio_ma.pop(0)

        select_ratio_ma.append(select_ratio)
        smoothed_select_ratio = np.mean(select_ratio_ma, axis=0)

        cov_mask = get_cov_mask(smoothed_select_ratio)

    return smoothed_select_ratio


def _prepare_feature_matrix(
    df: pd.DataFrame,
    selected_features: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Prepare feature matrix for CDS tests.

    If selected_features is empty, use a constant dummy feature. This corresponds
    to testing distribution shift conditioned on the empty set.
    """
    selected_features = list(selected_features)

    if len(selected_features) == 0:
        return np.zeros((len(df), 1), dtype=float), selected_features

    missing_features = [col for col in selected_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Selected features not found in dataframe: {missing_features}")

    return df[selected_features].values, selected_features


def _calculate_eval_weights(
    X_ref_train_scaled: np.ndarray,
    X_eval_scaled: np.ndarray,
    has_conditioning_features: bool,
) -> np.ndarray:
    """
    Calculate propensity weights for eval samples.

    If there are no conditioning features, return uniform weights.
    """
    if not has_conditioning_features:
        return np.ones(X_eval_scaled.shape[0], dtype=float)

    return calculate_propensity_weights(X_ref_train_scaled, X_eval_scaled)


def _stratify_if_possible(y: np.ndarray) -> np.ndarray | None:
    """
    Use stratified split only when every class has at least two samples.
    """
    labels, counts = np.unique(y, return_counts=True)

    if len(labels) > 1 and np.min(counts) >= 2:
        return y

    return None


# =============================================================================
# Phase 2: Stable variable selection
# =============================================================================

def stable_variable_selection(
    df: pd.DataFrame,
    args: Any,
    target_column: str,
    task: TaskType = "regression",
    prediction_model: PredictionModelType = "stg",
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Select stable variables for a target column.

    Parameters
    ----------
    df:
        Input dataframe containing both target and candidate features.

    args:
        Configuration object.

    target_column:
        Target variable whose stable predictors are selected.

    task:
        Either "regression" or "classification".

    prediction_model:
        One of "stg", "catboost", or "linear".

    Returns
    -------
    selected_feature_names:
        Names of selected stable features.

    feature_weights_df:
        DataFrame with columns ["feature", "weight"], sorted by weight.
    """
    if target_column not in df.columns:
        raise ValueError(f"target_column={target_column} is not in df.columns.")

    if task not in {"regression", "classification"}:
        raise ValueError(f"Unknown task: {task}")

    if prediction_model not in {"stg", "catboost", "linear"}:
        raise ValueError(f"Unknown prediction_model: {prediction_model}")

    X = df.drop(columns=[target_column])
    y = df[target_column].values
    feature_names = X.columns.to_numpy()

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=FEATURE_SELECTION_RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)

    # -------------------------------------------------------------------------
    # Regression
    # -------------------------------------------------------------------------
    if task == "regression":
        if prediction_model == "stg":
            feature_weights = _fit_stg_feature_weights(
                X_train_scaled,
                y_train,
                args,
            )

        elif prediction_model == "linear":
            sample_weights = _get_stable_weights(X_train_scaled, args)

            model = LinearRegression(fit_intercept=False)
            model.fit(
                X_train_scaled,
                y_train,
                sample_weight=sample_weights,
            )

            feature_weights = np.abs(model.coef_)

        elif prediction_model == "catboost":
            sample_weights = _get_stable_weights(X_train_scaled, args)

            model = CatBoostRegressor(
                verbose=False,
                iterations=3000,
                subsample=1.0,
                bootstrap_type="Bernoulli",
                learning_rate=0.05,
                depth=4,
                loss_function="RMSE",
            )

            model.fit(
                X_train_scaled,
                y_train,
                sample_weight=sample_weights,
            )

            feature_weights = model.feature_importances_

    # -------------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------------
    else:
        if prediction_model == "stg":
            raise NotImplementedError("STG is not supported for classification.")

        # If the target has only one class in the training split, feature
        # importance is not identifiable. Return all features with large weights.
        if len(np.unique(y_train)) < 2:
            feature_weights = 10.0 * np.ones(len(feature_names))
            feature_weights_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "weight": feature_weights,
                }
            )
            return feature_names, feature_weights_df

        sample_weights = _get_stable_weights(X_train_scaled, args)

        if prediction_model == "linear":
            model = LogisticRegression(
                fit_intercept=False,
                max_iter=1000,
            )

            model.fit(
                X_train_scaled,
                y_train,
                sample_weight=sample_weights,
            )

            feature_weights = np.mean(np.abs(model.coef_), axis=0)

        elif prediction_model == "catboost":
            binary_classification = len(np.unique(y_train)) == 2
            loss_function = "Logloss" if binary_classification else "MultiClass"

            model = CatBoostClassifier(
                verbose=False,
                iterations=1000,
                learning_rate=0.05,
                depth=3,
                loss_function=loss_function,
            )

            model.fit(
                X_train_scaled,
                y_train.astype(str),
                sample_weight=sample_weights,
            )

            feature_weights = model.feature_importances_

    selected_feature_names, feature_weights_df = _select_features_by_weight_ratio(
        feature_names=feature_names,
        feature_weights=feature_weights,
        args=args,
    )

    return selected_feature_names, feature_weights_df


# =============================================================================
# Phase 3: Conditional distribution shift tests
# =============================================================================

def classification_cds_test(
    df_ref: pd.DataFrame,
    df_eval: pd.DataFrame,
    target_column: str,
    selected_features: Sequence[str],
    default_large_score: float = 500.0,
) -> float:
    """
    Conditional distribution shift test for categorical targets.

    A classifier is trained on reference data to estimate P(Y | S), where S is
    the selected stable variable set. The RCA score compares prediction loss on
    evaluation data against held-out reference data.
    """
    if target_column not in df_ref.columns:
        raise ValueError(f"target_column={target_column} is not in df_ref.columns.")

    if target_column not in df_eval.columns:
        raise ValueError(f"target_column={target_column} is not in df_eval.columns.")

    X_ref, selected_features = _prepare_feature_matrix(df_ref, selected_features)
    X_eval, _ = _prepare_feature_matrix(df_eval, selected_features)

    has_conditioning_features = len(selected_features) > 0

    y_ref = df_ref[target_column].values
    y_eval = df_eval[target_column].values

    # If the reference target is single-class, the conditional mechanism is
    # degenerate. Keep the original behavior and return a large score.
    if len(np.unique(y_ref)) < 2:
        return float(default_large_score)

    all_labels = np.unique(np.concatenate([y_ref, y_eval]))
    label_to_int = {label: idx for idx, label in enumerate(all_labels)}

    y_ref_int = np.array([label_to_int[label] for label in y_ref])
    y_eval_int = np.array([label_to_int[label] for label in y_eval])

    stratify = _stratify_if_possible(y_ref_int)

    X_ref_train, X_ref_test, y_ref_train, y_ref_test = train_test_split(
        X_ref,
        y_ref_int,
        test_size=0.2,
        random_state=CDS_RANDOM_STATE,
        stratify=stratify,
    )

    if len(np.unique(y_ref_train)) < 2:
        return float(default_large_score)

    scaler = StandardScaler()
    X_ref_train_scaled = scaler.fit_transform(X_ref_train)
    X_ref_test_scaled = scaler.transform(X_ref_test)
    X_eval_scaled = scaler.transform(X_eval)

    train_labels = set(y_ref_train)
    eval_labels = set(y_eval_int)
    ref_test_labels = set(y_ref_test)

    # CatBoost cannot evaluate labels that were unseen during training.
    if len(eval_labels - train_labels) > 0:
        return float(default_large_score)

    if len(ref_test_labels - train_labels) > 0:
        return float(default_large_score)

    binary_target = len(train_labels) == 2
    loss_function = "Logloss" if binary_target else "MultiClass"

    model = CatBoostClassifier(
        loss_function=loss_function,
        verbose=False,
    )

    model.fit(X_ref_train_scaled, y_ref_train)

    eval_weights = _calculate_eval_weights(
        X_ref_train_scaled,
        X_eval_scaled,
        has_conditioning_features=has_conditioning_features,
    )

    y_ref_prob = model.predict_proba(X_ref_test_scaled)
    y_eval_prob = model.predict_proba(X_eval_scaled)

    class_labels = model.classes_

    ref_loss = log_loss(
        y_ref_test,
        y_ref_prob,
        labels=class_labels,
    )

    eval_loss = log_loss(
        y_eval_int,
        y_eval_prob,
        labels=class_labels,
        sample_weight=eval_weights,
    )

    rca_score = (eval_loss - ref_loss) / (ref_loss + EPS)
    return float(rca_score)


def regression_cds_test(
    df_ref: pd.DataFrame,
    df_eval: pd.DataFrame,
    target_column: str,
    selected_features: Sequence[str],
) -> float:
    """
    Conditional distribution shift test for continuous targets.

    A regressor is trained on reference data to estimate E[Y | S], where S is
    the selected stable variable set. The RCA score compares prediction error on
    evaluation data against held-out reference data.
    """
    if target_column not in df_ref.columns:
        raise ValueError(f"target_column={target_column} is not in df_ref.columns.")

    if target_column not in df_eval.columns:
        raise ValueError(f"target_column={target_column} is not in df_eval.columns.")

    X_ref, selected_features = _prepare_feature_matrix(df_ref, selected_features)
    X_eval, _ = _prepare_feature_matrix(df_eval, selected_features)

    has_conditioning_features = len(selected_features) > 0

    y_ref = df_ref[target_column].values
    y_eval = df_eval[target_column].values

    X_ref_train, X_ref_test, y_ref_train, y_ref_test = train_test_split(
        X_ref,
        y_ref,
        test_size=0.2,
        random_state=CDS_RANDOM_STATE,
    )

    feature_scaler = StandardScaler()
    X_ref_train_scaled = feature_scaler.fit_transform(X_ref_train)
    X_ref_test_scaled = feature_scaler.transform(X_ref_test)
    X_eval_scaled = feature_scaler.transform(X_eval)

    target_scaler = MinMaxScaler()
    y_ref_train_scaled = target_scaler.fit_transform(
        y_ref_train.reshape(-1, 1)
    ).reshape(-1)

    y_ref_test_scaled = target_scaler.transform(
        y_ref_test.reshape(-1, 1)
    ).reshape(-1)

    y_eval_scaled = target_scaler.transform(
        y_eval.reshape(-1, 1)
    ).reshape(-1)

    model = CatBoostRegressor(
        loss_function="RMSE",
        verbose=False,
    )

    model.fit(X_ref_train_scaled, y_ref_train_scaled)

    eval_weights = _calculate_eval_weights(
        X_ref_train_scaled,
        X_eval_scaled,
        has_conditioning_features=has_conditioning_features,
    )

    y_ref_pred = model.predict(X_ref_test_scaled)
    y_eval_pred = model.predict(X_eval_scaled)

    ref_mse = mean_squared_error(y_ref_test_scaled, y_ref_pred)
    eval_mse = mean_squared_error(
        y_eval_scaled,
        y_eval_pred,
        sample_weight=eval_weights,
    )

    rca_score = (eval_mse - ref_mse) / (ref_mse + EPS)
    return float(rca_score)