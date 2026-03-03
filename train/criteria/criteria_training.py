"""
Main training script for SJT criteria evaluation using DSPy.

Evaluates responses on multiple criteria using LLM-based evaluators with F1 optimization.
Supports OpenAI, Anthropic, and Google providers.
"""

import os
import sys
import json
import time
import logging
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict

import numpy as np

# Add root directory to path and load .env via config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROVIDERS, DATA_DIR, OPTIMIZE, USE_VALIDATION, RATE_LIMITS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup DSPy cache
_DSPY_CACHE_DIR = os.environ.setdefault(
    "DSPY_CACHEDIR",
    os.path.join(tempfile.gettempdir(), "dspy_cache")
)
os.makedirs(_DSPY_CACHE_DIR, exist_ok=True)

# Local imports add the file criteria_definitions.py with your criteira definitions and get_all_criteria function 
from criteria_definitions import CRITERIA_DEFINITIONS, get_all_criteria
from dspy_modules import setup_dspy, create_evaluator
from data_utils import load_quality_stratified_data, normalize_label
from metrics import calculate_metrics, f1_metric


def evaluate_criterion(criterion: str, train_examples: List, test_examples: List,
                       optimize: bool = True, max_bootstraps: int = 5,
                       provider: str = "openai") -> Dict:
    """Evaluate a single criterion with optional F1 optimization."""

    logger.info(f"Training samples: {len(train_examples)}")
    logger.info(f"Test samples: {len(test_examples)}")

    rate_limit_delay = RATE_LIMITS.get(provider, 0)
    if rate_limit_delay > 0:
        logger.info(f"Rate limit: {rate_limit_delay}s between requests for {provider}")

    # Create evaluator
    evaluator = create_evaluator(criterion)

    # Optimize with F1 metric if training data available
    if optimize and len(train_examples) >= 3:
        logger.info(f"Optimizing with F1 metric using {min(max_bootstraps, len(train_examples))} examples...")

        try:
            from dspy.teleprompt import BootstrapFewShot

            optimizer = BootstrapFewShot(
                metric=f1_metric,
                max_bootstrapped_demos=max_bootstraps,
                max_labeled_demos=max_bootstraps
            )

            evaluator = optimizer.compile(
                evaluator,
                trainset=train_examples[:max_bootstraps]
            )
            logger.info("Optimization complete")

        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            logger.info("Continuing with unoptimized evaluator")

    elif len(train_examples) == 0:
        logger.info("No training data - using zero-shot evaluation")

    # Run predictions
    predictions = []
    true_labels = []

    logger.info(f"Running predictions on {len(test_examples)} test examples...")

    last_request_time = 0

    for idx, example in enumerate(test_examples):
        if (idx + 1) % 10 == 0:
            logger.info(f"Progress: {idx + 1}/{len(test_examples)}")

        # Rate limiting
        if rate_limit_delay > 0:
            elapsed = time.time() - last_request_time
            if elapsed < rate_limit_delay:
                time.sleep(rate_limit_delay - elapsed)

        try:
            result = evaluator.forward(
                scenario=example.scenario,
                question=example.question,
                response=example.response,
                aspect=example.aspect
            )
            last_request_time = time.time()

            try:
                predicted_label = normalize_label(result.label, criterion)
                predictions.append(predicted_label)
                true_labels.append(example.majority_label)

            except ValueError as ve:
                logger.warning(f"Invalid label prediction: {result.label} - {ve}")
                continue

        except Exception as e:
            error_msg = str(e).lower()
            if "rate_limit" in error_msg or "rate limit" in error_msg:
                logger.warning("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                last_request_time = time.time()

                # Retry
                try:
                    result = evaluator.forward(
                        scenario=example.scenario,
                        question=example.question,
                        response=example.response,
                        aspect=example.aspect
                    )
                    predicted_label = normalize_label(result.label, criterion)
                    predictions.append(predicted_label)
                    true_labels.append(example.majority_label)
                except Exception as retry_error:
                    logger.error(f"Retry failed: {retry_error}")
                    continue
            else:
                logger.error(f"Error predicting: {e}")
                continue

    if not predictions:
        return None

    logger.info(f"Completed {len(predictions)}/{len(test_examples)} predictions")

    # Calculate metrics
    results = calculate_metrics(true_labels, predictions, criterion)
    results['criterion'] = criterion
    results['test_samples'] = len(test_examples)

    return results


def evaluate_all_criteria(data_dir: str, criteria_list: List[str],
                          optimize: bool = True, provider: str = "openai",
                          use_validation: bool = False) -> Dict:
    """Evaluate all criteria with F1 optimization."""

    test_split = 'val' if use_validation else 'test'
    criteria_results = {}

    for criterion in criteria_list:
        logger.info("=" * 60)
        logger.info(f"Evaluating: {criterion}")
        logger.info("=" * 60)

        try:
            train_examples = load_quality_stratified_data(data_dir, criterion, 'train')
            test_examples = load_quality_stratified_data(data_dir, criterion, test_split)

            if not test_examples:
                logger.warning(f"Skipping {criterion} - no {test_split} examples")
                continue

            if not train_examples:
                logger.warning(f"No training examples for {criterion} - using zero-shot")

            results = evaluate_criterion(
                criterion, train_examples, test_examples,
                optimize, provider=provider
            )

            if results:
                criteria_results[criterion] = results
                logger.info("Results:")
                logger.info(f"  Accuracy: {results['accuracy']:.3f}")
                logger.info(f"  Kappa: {results['kappa']:.3f}")
                logger.info(f"  F1 Score: {results['f1_score']:.3f}")
            else:
                logger.warning(f"No successful predictions for {criterion}")

        except Exception as e:
            logger.error(f"Error evaluating {criterion}: {e}")
            traceback.print_exc()
            continue

    return criteria_results


def print_comparison_table(all_provider_results: Dict):
    """Print comparison table across all providers."""

    logger.info("=" * 100)
    logger.info("MODEL COMPARISON - F1 OPTIMIZED")
    logger.info("=" * 100)

    logger.info("OVERALL PERFORMANCE:")
    logger.info(f"{'Provider':<25} {'Model':<35} {'Accuracy':>12} {'Kappa':>12} {'F1 Score':>12}")
    logger.info("-" * 100)

    for provider_key, data in all_provider_results.items():
        results = data['criteria_results']
        if results:
            avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
            avg_kappa = np.mean([r['kappa'] for r in results.values()])
            avg_f1 = np.mean([r['f1_score'] for r in results.values()])
            logger.info(f"{data['provider']:<25} {data['model']:<35} {avg_accuracy:>12.3f} {avg_kappa:>12.3f} {avg_f1:>12.3f}")


if __name__ == "__main__":
    CRITERIA_TO_EVALUATE = get_all_criteria()

    logger.info(f"Evaluating {len(CRITERIA_TO_EVALUATE)} criteria")
    logger.info(f"Using {'VALIDATION' if USE_VALIDATION else 'TEST'} set")

    all_provider_results = {}

    for provider_config in PROVIDERS:
        provider = provider_config["provider"]
        model = provider_config["model"]

        logger.info("=" * 100)
        logger.info(f"TESTING PROVIDER: {provider.upper()} - {model}")
        logger.info("=" * 100)

        setup_dspy(provider=provider, model=model, debug=True)

        criteria_results = evaluate_all_criteria(
            data_dir=DATA_DIR,
            criteria_list=CRITERIA_TO_EVALUATE,
            optimize=OPTIMIZE,
            provider=provider,
            use_validation=USE_VALIDATION
        )

        if criteria_results:
            all_provider_results[f"{provider}_{model}"] = {
                'provider': provider,
                'model': model,
                'criteria_results': criteria_results
            }

            # Save results
            eval_type = 'val' if USE_VALIDATION else 'test'
            results_file = f"results_{provider}_{model.replace('-', '_')}_{eval_type}.json"

            with open(results_file, 'w') as f:
                json_results = {
                    criterion: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                               for k, v in res.items()}
                    for criterion, res in criteria_results.items()
                }
                json.dump(json_results, f, indent=2)

            logger.info(f"Results saved to: {results_file}")

    if len(all_provider_results) > 1:
        print_comparison_table(all_provider_results)
