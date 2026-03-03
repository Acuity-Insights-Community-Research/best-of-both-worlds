"""
Evaluation metrics and statistical analysis for score prediction.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from scipy import stats
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import SCORE_RANGE

logger = logging.getLogger(__name__)

# Optional statsmodels imports
try:
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.power import TTestIndPower
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("statsmodels not installed - multiple comparisons and power analysis will be skipped.")


def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=95) -> Tuple[float, float, float]:
    """
    Bootstrap confidence intervals for any scalar metric.

    Returns: mean, lower, upper
    """
    rng = np.random.default_rng(42)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue

    lower = float(np.percentile(scores, (100 - ci) / 2))
    upper = float(np.percentile(scores, 100 - (100 - ci) / 2))
    return float(np.mean(scores)), lower, upper


def ordinal_accuracy(y_true, y_pred, tolerance=1) -> float:
    """Calculate accuracy within tolerance."""
    return np.mean(np.abs(y_true - y_pred) <= tolerance)


def evaluate_model(y_true, y_pred, model_name: str = "Model") -> Dict:
    """
    Full evaluation with bootstrapped 95% CIs and balanced sample weights.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_clipped = np.clip(y_pred, SCORE_RANGE[0], SCORE_RANGE[1])
    y_pred_rounded = np.round(y_pred_clipped)

    # Sample weights correcting for extreme score oversampling
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_true_arr)

    # Point estimates
    mse = mean_squared_error(y_true_arr, y_pred_clipped)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_arr, y_pred_clipped)
    r2 = r2_score(y_true_arr, y_pred_clipped)
    mean_error = np.mean(y_pred_clipped - y_true_arr)

    exact_acc = np.mean(y_true_arr == y_pred_rounded)
    within_1_acc = ordinal_accuracy(y_true_arr, y_pred_rounded, tolerance=1)
    within_2_acc = ordinal_accuracy(y_true_arr, y_pred_rounded, tolerance=2)

    # Weighted metrics (corrected for class imbalance)
    weighted_exact_acc = float(np.average(y_true_arr == y_pred_rounded, weights=sample_weights))
    weighted_within_1_acc = float(np.average(np.abs(y_true_arr - y_pred_rounded) <= 1, weights=sample_weights))

    # Bootstrap 95% CIs
    rmse_mean, rmse_lo, rmse_hi = bootstrap_metric(
        y_true_arr, y_pred_clipped, lambda a, b: np.sqrt(mean_squared_error(a, b))
    )
    mae_mean, mae_lo, mae_hi = bootstrap_metric(
        y_true_arr, y_pred_clipped, lambda a, b: mean_absolute_error(a, b)
    )
    exact_mean, exact_lo, exact_hi = bootstrap_metric(
        y_true_arr, y_pred_rounded, lambda a, b: np.mean(a == b)
    )
    within1_mean, within1_lo, within1_hi = bootstrap_metric(
        y_true_arr, y_pred_rounded, lambda a, b: np.mean(np.abs(a - b) <= 1)
    )

    logger.info(f"{model_name} Performance:")
    logger.info(f"  RMSE:              {rmse:.4f}  (95% CI: {rmse_lo:.4f}–{rmse_hi:.4f})")
    logger.info(f"  MAE:               {mae:.4f}  (95% CI: {mae_lo:.4f}–{mae_hi:.4f})")
    logger.info(f"  R²:                {r2:.4f}")
    logger.info(f"  Mean Error (bias): {mean_error:.4f}")
    logger.info(f"  Exact Accuracy:    {exact_acc:.4f}  (95% CI: {exact_lo:.4f}–{exact_hi:.4f})")
    logger.info(f"  Exact Acc (wtd):   {weighted_exact_acc:.4f}")
    logger.info(f"  ±1 Accuracy:       {within_1_acc:.4f}  (95% CI: {within1_lo:.4f}–{within1_hi:.4f})")
    logger.info(f"  ±1 Acc (wtd):      {weighted_within_1_acc:.4f}")
    logger.info(f"  ±2 Accuracy:       {within_2_acc:.4f}")

    # Detailed analyses
    cm_df = create_confusion_matrix_table(y_true_arr, y_pred_clipped)
    scale_analysis = analyze_scale_utilization(y_true_arr, y_pred_clipped)

    return {
        'rmse': rmse, 'rmse_ci': (rmse_lo, rmse_hi),
        'mae': mae, 'mae_ci': (mae_lo, mae_hi),
        'r2': r2,
        'mean_error': mean_error,
        'exact_accuracy': exact_acc, 'exact_accuracy_ci': (exact_lo, exact_hi),
        'exact_accuracy_weighted': weighted_exact_acc,
        'within_1_accuracy': within_1_acc, 'within_1_accuracy_ci': (within1_lo, within1_hi),
        'within_1_accuracy_weighted': weighted_within_1_acc,
        'within_2_accuracy': within_2_acc,
        'predictions': y_pred_rounded,
        'confusion_matrix': cm_df,
        'scale_analysis': scale_analysis
    }


def create_confusion_matrix_table(y_true, y_pred) -> pd.DataFrame:
    """Create confusion matrix showing actual vs predicted scores."""
    y_pred_rounded = np.round(y_pred).astype(int)
    y_true_int = np.asarray(y_true).astype(int)

    all_scores = sorted(set(y_true_int) | set(y_pred_rounded))
    cm = confusion_matrix(y_true_int, y_pred_rounded, labels=all_scores)
    cm_df = pd.DataFrame(cm, index=all_scores, columns=all_scores)

    logger.info("=" * 60)
    logger.info("CONFUSION MATRIX: Actual (rows) vs Predicted (columns)")
    logger.info("=" * 60)
    logger.info(f"\n{cm_df.to_string()}")

    logger.info("=" * 60)
    logger.info("PER-SCORE ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"{'Score':<8} {'Count':<8} {'Correct':<10} {'Accuracy':<10} {'Avg Error':<12}")
    logger.info("-" * 60)

    for score in all_scores:
        mask = y_true_int == score
        if mask.sum() == 0:
            continue
        count = mask.sum()
        correct = (y_pred_rounded[mask] == score).sum()
        accuracy = correct / count * 100
        avg_error = np.mean(y_pred_rounded[mask] - score)
        logger.info(f"{score:<8} {count:<8} {correct:<10} {accuracy:>6.2f}%    {avg_error:>+6.2f}")

    return cm_df


def analyze_scale_utilization(y_true, y_pred) -> Dict:
    """Analyze whether the full score scale is being utilized."""
    y_pred_rounded = np.round(y_pred).astype(int)
    y_pred_clipped = np.clip(y_pred_rounded, SCORE_RANGE[0], SCORE_RANGE[1])
    y_true_array = np.asarray(y_true).astype(int)
    y_pred_array = np.asarray(y_pred_clipped).astype(int)

    full_scale = list(range(SCORE_RANGE[0], SCORE_RANGE[1] + 1))
    actual_scores = sorted(np.unique(y_true_array))
    predicted_scores = sorted(np.unique(y_pred_array))

    logger.info("=" * 60)
    logger.info("SCALE UTILIZATION ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Expected Score Range: {SCORE_RANGE[0]} - {SCORE_RANGE[1]}")
    logger.info(f"Actual Scores Present: {actual_scores}")
    logger.info(f"Predicted Scores Used: {list(predicted_scores)}")

    missing_in_predicted = set(full_scale) - set(predicted_scores)

    actual_mean = y_true_array.mean()
    pred_mean = y_pred.mean()
    bias = pred_mean - actual_mean

    actual_utilization = (len(actual_scores) / len(full_scale)) * 100
    pred_utilization = (len(predicted_scores) / len(full_scale)) * 100

    logger.info(f"Actual data uses: {actual_utilization:.1f}% of scale")
    logger.info(f"Model uses: {pred_utilization:.1f}% of scale")

    return {
        'actual_scores': actual_scores,
        'predicted_scores': list(predicted_scores),
        'missing_in_predicted': sorted(missing_in_predicted),
        'actual_mean': actual_mean,
        'predicted_mean': pred_mean,
        'bias': bias,
        'actual_utilization': actual_utilization,
        'predicted_utilization': pred_utilization
    }


def run_multiple_comparison_correction(all_results: Dict, y_tests) -> Dict:
    """Apply Benjamini-Hochberg FDR correction across model comparisons."""
    if not HAS_STATSMODELS:
        logger.warning("Skipping multiple comparisons correction - statsmodels not installed")
        return {}

    logger.info("=" * 60)
    logger.info("MULTIPLE COMPARISONS CORRECTION (Benjamini-Hochberg FDR)")
    logger.info("=" * 60)

    y_true = np.asarray(y_tests)
    model_names = list(all_results.keys())
    p_values = []
    comparison_labels = []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a = model_names[i]
            name_b = model_names[j]
            err_a = np.abs(y_true - all_results[name_a]['predictions'])
            err_b = np.abs(y_true - all_results[name_b]['predictions'])
            _, p = stats.ttest_rel(err_a, err_b)
            p_values.append(p)
            comparison_labels.append(f"{name_a} vs {name_b}")

    rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')

    logger.info(f"{'Comparison':<35} {'p (raw)':>10} {'p (BH-corrected)':>18} {'Significant':>12}")
    logger.info("-" * 78)
    for label, p_raw, p_corr, sig in zip(comparison_labels, p_values, p_corrected, rejected):
        sig_str = "Yes *" if sig else "No"
        logger.info(f"{label:<35} {p_raw:>10.4f} {p_corr:>18.4f} {sig_str:>12}")

    return {
        'labels': comparison_labels,
        'p_raw': p_values,
        'p_corrected': list(p_corrected),
        'rejected': list(rejected)
    }


def run_power_analysis(n_train: int, n_test: int, observed_effect_size: float = None, alpha: float = 0.05) -> Dict:
    """Post-hoc power analysis."""
    if not HAS_STATSMODELS:
        logger.warning("Skipping power analysis - statsmodels not installed")
        return {}

    logger.info("=" * 60)
    logger.info("POST-HOC POWER ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Training n={n_train}, Test n={n_test}, α={alpha}")

    analysis = TTestIndPower()
    results = {}

    effect_sizes = [0.2, 0.5, 0.8]
    if observed_effect_size is not None:
        effect_sizes = [observed_effect_size] + effect_sizes

    logger.info(f"{'Effect Size':<25} {'Achieved Power':>15} {'Min n for 80% power':>22}")
    logger.info("-" * 65)

    for es in effect_sizes:
        label = f"observed (d={es:.3f})" if (observed_effect_size is not None and es == observed_effect_size) else f"d={es:.1f}"
        power = analysis.solve_power(effect_size=es, nobs1=n_test, alpha=alpha)
        min_n = analysis.solve_power(effect_size=es, power=0.8, alpha=alpha)
        logger.info(f"{label:<25} {power:>15.3f} {int(np.ceil(min_n)):>22}")
        results[es] = {'power': float(power), 'min_n_for_80pct': int(np.ceil(min_n))}

    return results


def get_feature_importance(model, feature_names, top_n: int = 15) -> pd.DataFrame:
    """Extract feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        logger.info(f"Top {top_n} Feature Importances:")
        logger.info("-" * 70)
        for i, idx in enumerate(indices, 1):
            feat_name = feature_names[idx].replace('_encoded', '').replace('_weighted', ' (W)')
            logger.info(f"{i:2d}. {feat_name:50s} {importances[idx]:.4f}")

        return pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': [importances[i] for i in indices]
        })
    return None
