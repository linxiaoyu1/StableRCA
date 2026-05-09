import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

def calculate_propensity_weights(X_source, X_target, classifier_type='logistic', calibrate=False):
    """
    Calculate weights for source samples to mitigate covariate shift using propensity scores.
    
    The weight for a source sample x is w(x) = P(target|x) / P(source|x).
    We train a classifier to distinguish between source and target data.
    
    Args:
        X_source (np.ndarray): Features from the source (training) distribution.
        X_target (np.ndarray): Features from the target (test) distribution.
        classifier_type (str): Type of classifier to use ('logistic' or 'rf').
        calibrate (bool): Whether to calibrate the classifier probabilities.
        
    Returns:
        np.ndarray: Weights for each sample in X_source.
    """
    n_source = X_source.shape[0]
    n_target = X_target.shape[0]
    
    # Combine data
    X = np.vstack([X_source, X_target])
    # Labels: 1 for source, 0 for target
    y = np.hstack([np.ones(n_source), np.zeros(n_target)])
    
    # Initialize classifier with class_weight='balanced' to handle sample imbalance
    if classifier_type == 'logistic':
        clf = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    else:
        raise ValueError(f"Unsupported classifier_type: {classifier_type}")
        
    if calibrate:
        clf = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
        
    # Train classifier to distinguish source from target
    clf.fit(X, y)
    
    # Get probabilities for the target samples (the ones we want to reweight)
    # clf.predict_proba returns [P(y=0|x), P(y=1|x)] where y=0 is target, y=1 is source
    probs = clf.predict_proba(X_target)
    p_source = probs[:, 1]
    p_target = probs[:, 0]
    
    # Avoid division by zero
    p_source = np.clip(p_source, 1e-6, 1 - 1e-6)
    p_target = np.clip(p_target, 1e-6, 1 - 1e-6)
    
    # Calculate weights: w(x) = P(source|x) / P(target|x)
    # This reweights the target distribution to match the source distribution
    weights = p_source / p_target
    
    # Normalize weights so they mean 1
    weights = weights / np.mean(weights)
    
    return weights
