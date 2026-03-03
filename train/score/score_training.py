"""
Score Prediction Training Pipeline.
Combines LLM-based criteria predictions with traditional ML for score prediction.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add root directory to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    SCORE_CSV_FILE, SCORE_TEST_SIZE, SCORE_OUTPUT_DIR,
    SCORE_SAVE_TEST_RESULTS, SCORE_FEATURE_COLUMNS
)
from models import train_models, save_model
from evaluation import (
    run_multiple_comparison_correction,
    run_power_analysis,
    get_feature_importance
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(csv_file: str, test_size: float = 0.2,
                 save_test_results: bool = True, output_dir: str = './results') -> Dict:
    """
    Complete score prediction pipeline.

    Includes:
      - Scenario-specific weighted feature engineering
      - Gradient Boosting and Extra Trees training
      - Bootstrapped 95% CIs on all metrics
      - Multiple comparisons correction
      - Post-hoc power analysis
    """

    logger.info("=" * 60)
    logger.info("SCORE PREDICTION TRAINING PIPELINE")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(csv_file)
    logger.info(f"Dataset shape: {df.shape}")

    # Feature engineering
    logger.info("Engineering weighted features...")
    df_processed = engineer_weighted_features(df)

    # Prepare features
    logger.info("Preparing training data...")
    X = df_processed[SCORE_FEATURE_COLUMNS].copy()
    y = df_processed['score'].copy()

    # Preserve ID columns if present
    id_columns = []
    if 'response_id' in df_processed.columns:
        id_columns.append('response_id')
    if 'question_id' in df_processed.columns:
        id_columns.append('question_id')

    # Clean data
    mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[mask]
    y_clean = y[mask]
    ids_clean = df_processed.loc[mask, id_columns].copy() if id_columns else None

    logger.info(f"Clean samples: {len(X_clean)}")
    logger.info(f"Features: {len(SCORE_FEATURE_COLUMNS)}")
    logger.info(f"Target range: [{y_clean.min()}, {y_clean.max()}]")

    # Score distribution
    logger.info("Score Distribution:")
    for score, count in y_clean.value_counts().sort_index().items():
        pct = (count / len(y_clean)) * 100
        logger.info(f"  Score {int(score)}: {count:3d} samples ({pct:5.2f}%)")

    # Train/test split with stratification
    logger.info("Splitting data with stratification...")
    if ids_clean is not None:
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X_clean, y_clean, ids_clean,
            test_size=test_size,
            random_state=42,
            stratify=y_clean
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean,
            test_size=test_size,
            random_state=42,
            stratify=y_clean
        )
        ids_test = None

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    # Train models
    logger.info("Training models...")
    results = train_models(X_train, X_test, y_train, y_test)

    # Multiple comparisons correction
    logger.info("Running multiple comparisons correction...")
    mc_results = run_multiple_comparison_correction(results, y_test)

    # Power analysis
    logger.info("Running power analysis...")
    err_gb = np.abs(np.asarray(y_test) - results['gradient_boosting']['predictions'])
    err_et = np.abs(np.asarray(y_test) - results['extra_trees']['predictions'])
    pooled_std = np.sqrt((err_gb.std() ** 2 + err_et.std() ** 2) / 2)
    cohens_d = (abs(err_gb.mean() - err_et.mean()) / pooled_std) if pooled_std > 0 else 0.5
    logger.info(f"Observed Cohen's d (GB vs ET errors): {cohens_d:.4f}")

    power_results = run_power_analysis(
        n_train=len(X_train),
        n_test=len(X_test),
        observed_effect_size=cohens_d
    )

    # Feature importance
    best_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = results[best_name]['model']
    logger.info(f"Best model: {best_name} (R² = {results[best_name]['r2']:.3f})")
    get_feature_importance(best_model, SCORE_FEATURE_COLUMNS)

    # Save test results
    if save_test_results and ids_test is not None:
        logger.info("Saving test results...")
        os.makedirs(output_dir, exist_ok=True)

        for model_name, model_results in results.items():
            test_results_df = ids_test.copy()
            test_results_df['actual_score'] = y_test.values
            test_results_df['predicted_score'] = model_results['predictions']
            test_results_df['prediction_error'] = test_results_df['predicted_score'] - test_results_df['actual_score']
            test_results_df['absolute_error'] = test_results_df['prediction_error'].abs()

            output_file = os.path.join(output_dir, f'{model_name}_test_results.csv')
            test_results_df.to_csv(output_file, index=False)
            logger.info(f"Saved {model_name} results to: {output_file}")

    # Save best model
    model_output_path = os.path.join(output_dir, f'{best_name}_model.joblib')
    save_model(best_model, SCORE_FEATURE_COLUMNS, model_output_path)

    # Summary
    logger.info("=" * 60)
    logger.info("FINAL MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<30} {'Gradient Boosting':>20} {'Extra Trees':>20}")
    logger.info("-" * 73)

    for metric_label, key in [
        ("RMSE", "rmse"),
        ("MAE", "mae"),
        ("Exact Accuracy", "exact_accuracy"),
        ("±1 Accuracy", "within_1_accuracy"),
        ("R²", "r2"),
    ]:
        gb_val = results['gradient_boosting'].get(key, float('nan'))
        et_val = results['extra_trees'].get(key, float('nan'))
        logger.info(f"{metric_label:<30} {gb_val:>20.4f} {et_val:>20.4f}")

    return {
        'results': results,
        'multiple_comparisons': mc_results,
        'power_analysis': power_results,
        'cohens_d': cohens_d,
        'best_model': best_name
    }


def main():
    parser = argparse.ArgumentParser(description="Train score prediction models")
    parser.add_argument("--csv-file", default=SCORE_CSV_FILE, help="Path to features CSV")
    parser.add_argument("--test-size", type=float, default=SCORE_TEST_SIZE)
    parser.add_argument("--output-dir", default=SCORE_OUTPUT_DIR)
    parser.add_argument("--no-save", action="store_true", help="Don't save test results")

    args = parser.parse_args()

    run_pipeline(
        csv_file=args.csv_file,
        test_size=args.test_size,
        save_test_results=not args.no_save,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
