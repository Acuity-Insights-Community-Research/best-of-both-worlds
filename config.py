"""
Configuration loader for SJT AI Rating System.
Loads settings from .env file, config.yaml, and environment variables.
Priority: Environment variables > .env file > config.yaml > defaults
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    project_root = Path(__file__).parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try loading from current directory as fallback
        load_dotenv()
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass


# =============================================================================
# LLM PROVIDERS CONFIGURATION
# =============================================================================

PROVIDERS = [
    {"provider": "openai", "model": "gpt-4o-mini"},
    # {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
    # {"provider": "google", "model": "gemini-2.5-flash"}
]

# =============================================================================
# CRITERIA TRAINING CONFIGURATION
# =============================================================================

DATA_DIR = "/path/to/training_dataset"  # Update with your data directory
OPTIMIZE = True
USE_VALIDATION = False

# Rate limits by provider (seconds between requests)
RATE_LIMITS = {
    "anthropic": 5,  # 5 requests per minute
    "openai": 100,    # Tier 5 has high limits
    "google": 1       # Conservative delay
}

# =============================================================================
# SCORE TRAINING CONFIGURATION
# =============================================================================

SCORE_CSV_FILE = "/path/to/content_quality_features.csv"  # Update with your data
SCORE_TEST_SIZE = 0.2
SCORE_OUTPUT_DIR = "./results"
SCORE_SAVE_TEST_RESULTS = True
SCORE_RANGE = (1, 9)

# Gradient Boosting hyperparameters
GB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "subsample": 0.8,
    "loss": "huber",
    "alpha": 0.9,
    "random_state": 42
}

# Extra Trees hyperparameters
ET_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1
}

# Feature columns for score prediction add here

# Ordinal mappings for criteria encoding add here

# Scenario-specific base weights add here


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from .env file, YAML file, and environment variables.
    
    Priority order (highest to lowest):
    1. Environment variables (set in shell)
    2. .env file (loaded via python-dotenv)
    3. config.yaml file
    4. Default values
    
    Args:
        config_path: Path to config.yaml file. If None, looks for config.yaml in project root.
    
    Returns:
        Dictionary of configuration values.
    """
    if config_path is None:
        # Look for config.yaml in the project root (parent of this file)
        project_root = Path(__file__).parent
        config_path = project_root / "config.yaml"
    
    config = {}
    
    # Load from YAML file if it exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        print(f"Warning: Config file not found at {config_path}. Using defaults and environment variables.")
    
    # Environment variables take precedence
    env_config = {
        # API Keys
        'api_keys': {
            'anthropic': os.getenv('ANTHROPIC_API_KEY', config.get('api_keys', {}).get('anthropic')),
            'openai': os.getenv('OPENAI_API_KEY', config.get('api_keys', {}).get('openai')),
            'google': os.getenv('GOOGLE_API_KEY', config.get('api_keys', {}).get('google')),
        },
        
        # Criteria Training
        'criteria_training': {
            'base_filename': os.getenv('SJT_BASE_FILENAME', config.get('criteria_training', {}).get('base_filename')),
            'provider': os.getenv('CRITERIA_PROVIDER', config.get('criteria_training', {}).get('provider', 'anthropic')),
            'model_name': os.getenv('CRITERIA_MODEL_NAME', config.get('criteria_training', {}).get('model_name')),
            'sample_size': int(os.getenv('CRITERIA_SAMPLE_SIZE', str(config.get('criteria_training', {}).get('sample_size', 100)))),
            'max_samples': int(os.getenv('CRITERIA_MAX_SAMPLES', str(config.get('criteria_training', {}).get('max_samples', 15)))),
            'optimize': os.getenv('CRITERIA_OPTIMIZE', str(config.get('criteria_training', {}).get('optimize', True))).lower() == 'true',
            'optimizer_type': os.getenv('CRITERIA_OPTIMIZER', config.get('criteria_training', {}).get('optimizer_type', 'bootstrap')),
            'use_curriculum': os.getenv('CRITERIA_USE_CURRICULUM', str(config.get('criteria_training', {}).get('use_curriculum', False))).lower() == 'true',
            'curriculum_threshold': float(os.getenv('CRITERIA_CURRICULUM_THRESHOLD', str(config.get('criteria_training', {}).get('curriculum_threshold', 0.8)))),
            'model_dir': os.getenv('CRITERIA_MODEL_DIR', config.get('criteria_training', {}).get('model_dir')),
            'save_results': os.getenv('CRITERIA_SAVE_RESULTS', str(config.get('criteria_training', {}).get('save_results', True))).lower() == 'true',
        },
        
        # Score Training
        'score_training': {
            'csv_file': os.getenv('CONTENT_FEATURES_CSV', config.get('score_training', {}).get('csv_file')),
            'output_path': os.getenv('SCORE_MODEL_PATH', config.get('score_training', {}).get('output_path')),
            'test_size': float(os.getenv('SCORE_TEST_SIZE', str(config.get('score_training', {}).get('test_size', 0.2)))),
        },
        
        # Inference
        'inference': {
            'criteria_model_dir': os.getenv('CRITERIA_MODEL_DIR', config.get('inference', {}).get('criteria_model_dir')),
            'score_model_path': os.getenv('SCORE_MODEL_PATH', config.get('inference', {}).get('score_model_path')),
            'provider': os.getenv('INFERENCE_PROVIDER', config.get('inference', {}).get('provider', 'anthropic')),
            'model_name': os.getenv('INFERENCE_MODEL_NAME', config.get('inference', {}).get('model_name')),
        },
        
        # DSPy Cache
        'dspy_cache': {
            'cache_dir': os.getenv('DSPY_CACHEDIR', config.get('dspy_cache', {}).get('cache_dir')),
        },
    }
    
    return env_config


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a specific provider."""
    config = load_config()
    return config['api_keys'].get(provider.lower())


def get_criteria_config() -> Dict[str, Any]:
    """Get criteria training configuration."""
    config = load_config()
    return config['criteria_training']


def get_score_config() -> Dict[str, Any]:
    """Get score training configuration."""
    config = load_config()
    return config['score_training']


def get_inference_config() -> Dict[str, Any]:
    """Get inference configuration."""
    config = load_config()
    return config['inference']

