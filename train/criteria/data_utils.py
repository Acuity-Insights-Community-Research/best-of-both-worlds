"""
Data loading and processing utilities for SJT evaluation.
"""

import os
import logging
from typing import List
from collections import Counter
import dspy
import pandas as pd
# Local imports add the file criteria_definitions.py with your criteira definitions and get_all_criteria function 
from criteria_definitions import CRITERIA_DEFINITIONS, is_multi_label_criterion

logger = logging.getLogger(__name__)


def normalize_label(label: str, criterion: str) -> str:
    """Normalize label format for consistent comparison."""
    label = label.strip()

    # Multi-label criteria - return as-is
    if is_multi_label_criterion(criterion):
        return label

    label_upper = label.upper()

    criterion_labels = {
        lbl.strip().upper(): lbl
        for lbl in CRITERIA_DEFINITIONS[criterion]['labels']
    }

    normalized = criterion_labels.get(label_upper, None)

    if normalized is None:
        if label_upper in ['NOT APPLICABLE', 'NOT APPLICABLE FOR THIS QUESTION', 'N/A', 'NA']:
            raise ValueError(f"Label '{label}' is not valid for criterion {criterion}")
        raise ValueError(f"Unknown label '{label}' for criterion {criterion}")

    return normalized


def calculate_majority_label(rater_labels: List[str]) -> str:
    """Get majority label from multiple raters."""
    label_counts = Counter(rater_labels)
    return label_counts.most_common(1)[0][0]


def load_quality_stratified_data(data_dir: str, criterion: str, split: str = 'train') -> List:
    """
    Load quality-stratified data for a specific criterion.

    Args:
        data_dir: Directory containing train/val/test files
        criterion: Criterion to filter by
        split: 'train', 'val', or 'test'

    Returns:
        List of dspy.Example objects
    """
    split_file = os.path.join(data_dir, f"{split}_quality_stratified.csv")

    if not os.path.exists(split_file):
        logger.warning(f"{split.upper()} file not found: {split_file}")
        return []

    df = pd.read_csv(split_file)

    if 'criteria_id' not in df.columns:
        logger.error(f"'criteria_id' column not found in {split} file")
        return []

    df_criterion = df[df['criteria_id'] == criterion].copy()

    if len(df_criterion) == 0:
        logger.warning(f"No data found for criterion {criterion} in {split} file")
        return []

    # Add any special cases here if needed (e.g., for specific criteria with known issues)

    logger.info(f"Loaded {len(df_criterion)} rows for {criterion} from {split} file")

    examples = _create_examples(df_criterion, criterion)
    logger.info(f"Created {len(examples)} unique {split} examples")

    return examples


def _create_examples(df: pd.DataFrame, criterion: str) -> List:
    """Create dspy.Example objects from dataframe."""

    if 'response_question_id' not in df.columns:
        if 'question_id' in df.columns:
            df['response_question_id'] = df['response_id'] + '_' + df['question_id']
        else:
            df['response_question_id'] = df['response_id']

    grouped = df.groupby('response_question_id')
    examples = []

    for response_q_id, group in grouped:
        rater_labels = group['label'].tolist()
        majority_label = calculate_majority_label(rater_labels)

        first_row = group.iloc[0]

        # Get response text based on question
        question_id = first_row.get('question_id', 'Q1')
        if question_id == 'Q2':
            response_text = first_row.get('response_q2', '')
        else:
            response_text = first_row.get('response_q1', '')

        # Skip empty responses
        if pd.isna(response_text) or str(response_text).strip() == '' or str(response_text).lower() == 'nan':
            continue

        aspect = str(first_row.get('aspect_primary'))

        example = dspy.Example(
            response_id=str(first_row.get('response_id', '')),
            response_question_id=response_q_id,
            scenario=str(first_row.get('statement', first_row.get('scenario', ''))),
            question=str(first_row['question_text']),
            response=str(response_text),
            aspect=aspect,
            criterion_name=criterion,
            majority_label=majority_label
        ).with_inputs('scenario', 'question', 'response', 'aspect')

        examples.append(example)

    return examples
