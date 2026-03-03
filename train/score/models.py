"""
Model training for score prediction.
"""

import os
import logging
from typing import Dict, List, Tuple
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import GB_PARAMS, ET_PARAMS

from evaluation import evaluate_model

logger = logging.getLogger(__name__)


def train_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series) -> Dict:
    """Train Gradient Boosting and Extra Trees models."""

    results = {}

    logger.info("=" * 60)
    logger.info("Training Gradient Boosting...")
    logger.info("=" * 60)

    gb_model = GradientBoostingRegressor(**GB_PARAMS)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    results['gradient_boosting'] = evaluate_model(y_test, gb_pred, "Gradient Boosting")
    results['gradient_boosting']['model'] = gb_model

    logger.info("=" * 60)
    logger.info("Training Extra Trees...")
    logger.info("=" * 60)

    et_model = ExtraTreesRegressor(**ET_PARAMS)
    et_model.fit(X_train, y_train)
    et_pred = et_model.predict(X_test)
    results['extra_trees'] = evaluate_model(y_test, et_pred, "Extra Trees")
    results['extra_trees']['model'] = et_model

    return results


def save_model(model, feature_columns: List[str], output_path: str):
    """Save trained model and feature columns."""
    if output_path is None:
        raise ValueError("output_path must be provided to save the model.")

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload = {
        'model': model,
        'feature_columns': feature_columns
    }
    joblib.dump(payload, output_path)
    logger.info(f"Saved model to {output_path}")


def load_model(model_path: str) -> Tuple[object, List[str]]:
    """Load a persisted score model and feature ordering."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Score model file not found: {model_path}")

    payload = joblib.load(model_path)
    model = payload.get('model')
    feature_columns = payload.get('feature_columns')

    if model is None or feature_columns is None:
        raise ValueError("Model file is missing required keys 'model' or 'feature_columns'.")

    return model, feature_columns
