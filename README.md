# Best of Both Worlds: Combining LLMs and Traditional ML for automated scoring of an open-response situational judgment test

We have developed a system that can evaluate short text responses using LLM-based criteria evaluation and machine learning score prediction.
We share some of the training code below for research purposes. 

Note: This code cannot be used as is to develop your own model. It must be supplemented with context-dependent information such as criteria definitions, content, encodings for your data, etc. 


## Overview

This system evaluates responses across multiple criteria using DSPy-optimized LLM evaluators, then combines these predictions into a final score using a trained regression model.

### Key Features

- **Multi-criteria evaluation**: Evaluates responses across multiple criteria using LLM evaluators
- **DSPy optimization**: Automatically optimizes prompts and few-shot examples for each criterion
- **Score prediction**: Combines criteria predictions into final scores using three encdoding techniques

## Project Structure

```
ai_rating_research/
├── train/
│   ├── criteria/                      # LLM-based criteria evaluation (DSPy)
│   │   ├── criteria_training.py       # Main training script
│   │   ├── criteria_definitions.py    # YOUR CRITERIA (you must create this)
│   │   ├── data_utils.py              # Data loading utilities
│   │   ├── dspy_modules.py            # DSPy signatures and evaluators
│   │   └── metrics.py                 # F1, kappa, accuracy calculations
│   │
│   └── score/                         # Traditional ML score prediction
│       ├── score_training.py          # Main training pipeline
│       ├── evaluation.py              # Metrics, bootstrap CIs, power analysis
│       └── models.py                  # Gradient Boosting & Extra Trees
│
├── utils/
│   ├── __init__.py
│   └── logger.py                      # Logging utility
├── config.py                          # Central configuration (providers, hyperparams)
├── config.yaml.example                # Example configuration file
├── requirements.txt                   # Python dependencies
└── README.md
```

## Setup

### Prerequisites

- Python 3.9+
- Virtual environment (recommended)

### Installation

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Mac/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set API keys:**
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"  # For Anthropic Claude
   # OR
   export OPENAI_API_KEY="your-key-here"     # For OpenAI
   # OR
   export GOOGLE_API_KEY="your-key-here"     # For Google Gemini
   ```

4. **Configure settings:**
   ```bash
   cp config.yaml.example config.yaml
   # Edit config.yaml with your paths and settings
   ```

### Required User Files

You must create the following files with your own criteria and data:

#### 1. Criteria Definitions (`train/criteria/criteria_definitions.py`)

Create this file to define your evaluation criteria:

```python
# Example criteria_definitions.py

CRITERIA_DEFINITIONS = {
    "criterion_name": {
        "description": "What this criterion evaluates",
        "labels": ["Label1", "Label2", "Label3"],  # Possible labels for this criterion
    },
    # Add more criteria as needed
}

def get_all_criteria():
    """Return list of all criterion names to evaluate."""
    return list(CRITERIA_DEFINITIONS.keys())

def is_multi_label_criterion(criterion: str) -> bool:
    """Return True if criterion allows multiple labels per response."""
    # Implement based on your criteria
    return False
```

#### 2. Feature Columns and Mappings in config.py

Add your feature column names and encoding mappings to `config.py`:

```python
# Feature columns for score prediction
# These are the column names in your score training CSV that will be used as features
SCORE_FEATURE_COLUMNS = [
    "criterion_1_encoded",      # Ordinal-encoded criterion
    "criterion_2_encoded",
    "criterion_1_weighted",     # Weighted criterion (optional)
    "criterion_2_weighted",
    # ... list all your feature column names
]

# Ordinal mappings for criteria encoding
# Maps categorical labels to numeric values for ML models
# Order matters: arrange from lowest to highest quality/score
ORDINAL_MAPPINGS = {
    "criterion_1": {
        "Label1": 0,
        "Label2": 1,
        "Label3": 2,
        "Label4": 3
    },
    "criterion_2": {
        "Label1": 0,
        "Label2": 1,
        "Label3": 2
    },
    # Add mappings for each criterion in your CRITERIA_DEFINITIONS
}

# Scenario-specific weights (optional)
# Different scenarios may weight criteria differently
SCENARIO_WEIGHTS = {
    "scenario_type_A": {
        "criterion_1": 1.0,   # Full weight
        "criterion_2": 0.8,   # 80% weight
        "criterion_3": 0.5,   # 50% weight
    },
    "scenario_type_B": {
        "criterion_1": 0.7,
        "criterion_2": 1.0,
        "criterion_3": 0.9,
    },
    # Add weights for each scenario type in your data
}
```

#### 3. Feature Engineering Function (`train/score/score_training.py`)

You must implement the `engineer_weighted_features` function in `score_training.py`. This function transforms raw criteria labels into numeric features:

```python
def engineer_weighted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform criteria labels into numeric features for ML models.

    This function should:
    1. Apply ordinal encoding to convert labels to numbers
    2. Optionally apply scenario-specific weights
    3. Create any derived features (interactions, aggregations, etc.)

    Args:
        df: DataFrame with raw criteria labels

    Returns:
        DataFrame with engineered numeric features
    """
    from config import ORDINAL_MAPPINGS, SCENARIO_WEIGHTS

    df = df.copy()

    # Step 1: Ordinal encode each criterion
    for criterion, mapping in ORDINAL_MAPPINGS.items():
        if criterion in df.columns:
            df[f"{criterion}_encoded"] = df[criterion].map(mapping)

    # Step 2: Apply scenario-specific weights (optional)
    if 'scenario_type' in df.columns:
        for criterion, mapping in ORDINAL_MAPPINGS.items():
            if criterion in df.columns:
                df[f"{criterion}_weighted"] = df.apply(
                    lambda row: row[f"{criterion}_encoded"] *
                                SCENARIO_WEIGHTS.get(row['scenario_type'], {}).get(criterion, 1.0),
                    axis=1
                )

    # Step 3: Create aggregate features (optional)
    encoded_cols = [c for c in df.columns if c.endswith('_encoded')]
    if encoded_cols:
        df['criteria_mean'] = df[encoded_cols].mean(axis=1)
        df['criteria_sum'] = df[encoded_cols].sum(axis=1)

    return df
```

**Three encoding approaches** (as mentioned in the paper):

| Approach | Description | Use Case |
|----------|-------------|----------|
| **Ordinal** | Direct label-to-number mapping | When labels have clear ordering |
| **Weighted** | Ordinal × scenario weight | When criteria importance varies by context |
| **Binary** | Binary columns per label | When labels aren't ordinal |
| **Count-based** | Count based hen labels are counts |
## Data Format

### Criteria Training Data

Place your training data in a directory (set via `DATA_DIR` in `config.py`). The system expects quality-stratified CSV files:

```
your_data_directory/
├── train_quality_stratified.csv
├── val_quality_stratified.csv   # Optional, for validation
└── test_quality_stratified.csv
```

**Required columns:**

| Column | Description |
|--------|-------------|
| `response_id` | Unique identifier for each response |
| `criteria_id` | Name of the criterion being evaluated (must match keys in `CRITERIA_DEFINITIONS`) |
| `label` | Evaluation for this criterion |
| `question_id` | Question identifier (e.g., "Q1", "Q2") |
| `statement` | The scenario/situation statement |
| `question_text` | The question being answered |
| `response_q1` | Response text for question 1 |
| `response_q2` | Response text for question 2 |
| `aspect_primary` | Primary aspect being evaluated |

**Note:** Multiple rows per response/criterion are expected (one per rater). The system calculates majority labels automatically.

### Score Training Data

A single CSV file with aggregated criteria predictions and target scores.

**Required columns:**

| Column | Description |
|--------|-------------|
| `response_id` | Unique identifier for each response |
| `question_id` | Question identifier (optional) |
| `score` | Target score (integer, e.g., 1-9) |
| Criteria columns | One column per criterion with predicted labels |

## Training

### Step 1: Train Criteria Evaluators

Train LLM evaluators for each criterion using labeled data.

1. **Configure `config.py`:**
   ```python
   DATA_DIR = "/path/to/your/data_directory"
   PROVIDERS = [
       {"provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
   ]
   OPTIMIZE = True
   ```

2. **Run training:**
   ```bash
   python train/criteria/criteria_training.py
   ```

Results are saved as JSON files in the current directory.

### Step 2: Train Score Prediction Model

Train the regression model that converts criteria predictions to final scores.

1. **Configure `config.py`:**
   ```python
   SCORE_CSV_FILE = "/path/to/features.csv"
   SCORE_FEATURE_COLUMNS = ["criterion_1", "criterion_2", ...]
   SCORE_OUTPUT_DIR = "./results"
   ```

2. **Run training:**
   ```bash
   python train/score/score_training.py
   ```

   Or with command-line arguments:
   ```bash
   python train/score/score_training.py \
     --csv-file /path/to/features.csv \
     --output-dir ./results \
     --test-size 0.2
   ```

Models and test results are saved to the output directory.


## Quick Start Checklist

Before running the code, ensure you have:

- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Set your LLM API key as an environment variable
- [ ] Created `train/criteria/criteria_definitions.py` with your criteria
- [ ] Updated `config.py` with:
  - [ ] `DATA_DIR` pointing to your training data
  - [ ] `PROVIDERS` list with your chosen LLM provider(s)
  - [ ] `SCORE_CSV_FILE` path (for score training)
  - [ ] `SCORE_FEATURE_COLUMNS` list of your feature columns
- [ ] Prepared your training data CSVs in the required format

## License

This code is provided for research purposes. Please cite our paper if you use this work.

## Citation

```bibtex
@misc{walsh2026,
  title={Best of Both Worlds: Combining LLMs and Traditional ML for automated scoring of an open-response situational judgment test},
  author={Cole Walsh, Susha Suresh, Rodica Ivan},
  journal={Frontiers in Education},
  year={2026}
}
```

## Model Developers
Research Innovation Team @ Acuity Insights

## Documentation 
This README and our paper [LINK TBD]
