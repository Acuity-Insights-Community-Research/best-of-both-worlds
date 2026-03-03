"""
Metrics calculation for SJT evaluation.
Contains F1, kappa, and accuracy calculations.
"""

import numpy as np
from typing import List
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
# Local imports add the file criteria_definitions.py with your criteira definitions and get_all_criteria function AND BINARY VS MULTI-LABEL CRITEIRA
from criteria_definitions import (
    CRITERIA_DEFINITIONS,
    is_binary_criterion,
    is_multi_label_criterion
)


def calculate_multi_label_agreement(y_true: List[str], y_pred: List[str]) -> float:
    """
    Calculate agreement for multi-label criteria using Jaccard similarity.
    Used for OTHER_ASPECT_DEMONSTRATION.
    """
    agreements = []

    for true_label, pred_label in zip(y_true, y_pred):
        # Parse labels (might be comma-separated)
        if ',' in true_label:
            true_aspects = set([a.strip() for a in true_label.split(',')])
        else:
            true_aspects = {true_label.strip()}

        if ',' in pred_label:
            pred_aspects = set([a.strip() for a in pred_label.split(',')])
        else:
            pred_aspects = {pred_label.strip()}

        # Calculate Jaccard similarity
        union = true_aspects.union(pred_aspects)
        if len(union) == 0:
            agreements.append(1.0)
        else:
            intersection = true_aspects.intersection(pred_aspects)
            agreements.append(len(intersection) / len(union))

    return np.mean(agreements)


def calculate_quadratic_weighted_kappa(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    """Calculate quadratic weighted kappa for ordinal data."""
    y_true_numeric = [labels.index(label) for label in y_true]
    y_pred_numeric = [labels.index(label) for label in y_pred]

    return cohen_kappa_score(y_true_numeric, y_pred_numeric, weights='quadratic')


def calculate_metrics(true_labels: List[str], predictions: List[str], criterion: str) -> dict:
    """
    Calculate all relevant metrics for a criterion.

    Returns dict with: accuracy, kappa, f1_score
    """
    accuracy = accuracy_score(true_labels, predictions)

    results = {
        'accuracy': accuracy,
        'num_predictions': len(predictions)
    }

    labels = CRITERIA_DEFINITIONS[criterion]['labels']

    # Multi-label criteria use Jaccard similarity
    if is_multi_label_criterion(criterion):
        jaccard = calculate_multi_label_agreement(true_labels, predictions)
        results['jaccard'] = jaccard
        results['kappa'] = jaccard
        results['f1_score'] = jaccard

    # Binary criteria use Cohen's kappa
    elif is_binary_criterion(criterion):
        y_true_numeric = [labels.index(label) for label in true_labels]
        y_pred_numeric = [labels.index(label) for label in predictions]

        results['kappa'] = cohen_kappa_score(y_true_numeric, y_pred_numeric)
        results['f1_score'] = f1_score(y_true_numeric, y_pred_numeric, average='weighted', zero_division=0)

    # Ordinal criteria use quadratic weighted kappa
    else:
        results['kappa'] = calculate_quadratic_weighted_kappa(true_labels, predictions, labels)

        y_true_numeric = [labels.index(label) for label in true_labels]
        y_pred_numeric = [labels.index(label) for label in predictions]
        results['f1_score'] = f1_score(y_true_numeric, y_pred_numeric, average='weighted', zero_division=0)

    return results


def f1_metric(example, pred, trace=None):
    """F1 score metric for DSPy optimization."""
    from data_utils import normalize_label

    criterion = example.criterion_name

    # Multi-label handling
    if is_multi_label_criterion(criterion):
        true_label = example.majority_label
        try:
            pred_label = pred.label.strip()

            if ',' in true_label:
                true_aspects = set([a.strip() for a in true_label.split(',')])
            else:
                true_aspects = {true_label.strip()}

            if ',' in pred_label:
                pred_aspects = set([a.strip() for a in pred_label.split(',')])
            else:
                pred_aspects = {pred_label.strip()}

            union = true_aspects.union(pred_aspects)
            if len(union) == 0:
                return 1.0
            return len(true_aspects.intersection(pred_aspects)) / len(union)

        except (ValueError, AttributeError):
            return 0.0

    # Single-label handling
    labels = CRITERIA_DEFINITIONS[criterion]['labels']

    try:
        y_true = [labels.index(example.majority_label)]
        predicted_label = normalize_label(pred.label, criterion)
        y_pred = [labels.index(predicted_label)]
        return f1_score(y_true, y_pred, average='weighted', zero_division=0)
    except (ValueError, AttributeError):
        return 0.0
