"""
DSPy module definitions for SJT evaluation.
Contains signature and evaluator module classes.
"""

import os
import logging
import dspy
# Local imports add the file criteria_definitions.py with your criteira definitions and get_all_criteria function 
from criteria_definitions import CRITERIA_DEFINITIONS

logger = logging.getLogger(__name__)


def setup_dspy(provider: str = "anthropic", model: str = None, debug: bool = False):
    """Configure DSPy with API key from environment variables."""

    if provider == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        model_name = model or "gpt-4o-mini"

    elif provider == "anthropic":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        model_name = model or "claude-3-7-sonnet-20250219"

    elif provider == "google":
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        model_name = model or "gemini-2.5-flash"

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    lm = dspy.LM(
        model=model_name,
        max_tokens=1000,
        temperature=0.1
    )

    dspy.settings.configure(lm=lm)

    if debug:
        logger.info(f"DSPy configured with {provider} - {model_name}")

    return lm


class SJTEvaluatorSignature(dspy.Signature):
    """DSPy signature for evaluating SJT response on specific criterion."""

    scenario = dspy.InputField(desc="The scenario description")
    question = dspy.InputField(desc="The question being asked")
    response = dspy.InputField(desc="The user's response to evaluate")
    aspect = dspy.InputField(desc="The intended aspect being evaluated (for INTENDED_ASPECT_DEMONSTRATION) or general context")
    criterion_name = dspy.InputField(desc="Name of the criterion")
    criterion_description = dspy.InputField(desc="Description of what this criterion measures")
    valid_labels = dspy.InputField(desc="Valid labels for this criterion")

    label = dspy.OutputField(desc="The predicted label from the valid labels")
    reasoning = dspy.OutputField(desc="Brief explanation of the evaluation")


class SJTEvaluator(dspy.Module):
    """DSPy module for SJT criterion evaluation."""

    def __init__(self, criterion_name: str = None, criterion_description: str = None, valid_labels: list = None):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(SJTEvaluatorSignature)

        self.criterion_name = criterion_name
        self.criterion_description = criterion_description
        self.valid_labels = valid_labels

        # Add special handling notes for certain criteria
        
    def forward(self, scenario: str, question: str, response: str, aspect: str):
        """Run evaluation on a single response."""
        description = self.criterion_description + self.special_notes

        result = self.evaluate(
            scenario=scenario,
            question=question,
            response=response,
            aspect=aspect,
            criterion_name=self.criterion_name,
            criterion_description=description,
            valid_labels=", ".join(self.valid_labels)
        )
        return result


def create_evaluator(criterion: str) -> SJTEvaluator:
    """Factory function to create an evaluator for a specific criterion."""
    return SJTEvaluator(
        criterion_name=criterion,
        criterion_description=CRITERIA_DEFINITIONS[criterion]['description'],
        valid_labels=CRITERIA_DEFINITIONS[criterion]['labels']
    )
