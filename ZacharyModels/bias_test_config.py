"""
Configuration file for BERT bias testing.
Modify these settings to customize your bias testing setup.
"""

# Model configurations
BERT_MODELS = {
    "bert-base-uncased": {
        "model_class": "BertForMaskedLM",
        "model_path": "bert-base-uncased", 
        "description": "BERT Base - main encoder baseline",
        "expected_test_time": "15-30 minutes"
    },
    "roberta-base": {
        "model_class": "RobertaForMaskedLM",
        "model_path": "roberta-base",
        "description": "RoBERTa Base - strong comparative variant", 
        "expected_test_time": "15-30 minutes"
    },
    "google/electra-small-discriminator": {
        "model_class": "ElectraForMaskedLM",
        "model_path": "google/electra-small-discriminator",
        "description": "ELECTRA Small - efficient, diverse architecture",
        "expected_test_time": "10-20 minutes"
    },
    "bert-large-uncased": {
        "model_class": "BertForMaskedLM", 
        "model_path": "bert-large-uncased",
        "description": "BERT Large - scaled baseline",
        "expected_test_time": "30-45 minutes"
    }
}

# Bias types to evaluate
BIAS_TYPES = ["gender", "race", "religion"]

# SEAT test configurations
SEAT_TESTS = [
    "sent-weat1",   # Flowers vs insects with pleasant/unpleasant
    "sent-weat2",   # Instruments vs weapons with pleasant/unpleasant
    "sent-weat3",   # European vs African American names with pleasant/unpleasant
    "sent-weat4",   # European vs African American names with career/family
    "sent-weat5",   # European vs African American names with math/arts
    "sent-weat6",   # Male vs female names with career/family
    "sent-weat7",   # Math vs arts with male/female names
    "sent-weat8",   # Science vs arts with male/female names
]

# Test configurations
TEST_CONFIGS = {
    "quick": {
        "tests": ["stereoset"],
        "models": ["bert-base-uncased", "roberta-base"],
        "description": "Quick bias assessment using StereoSet"
    },
    "comprehensive": {
        "tests": ["stereoset", "crows", "seat"],
        "models": list(BERT_MODELS.keys()),
        "description": "Full bias evaluation across all benchmarks"
    },
    "baseline": {
        "tests": ["stereoset", "crows"],
        "models": ["bert-base-uncased"],
        "description": "Baseline BERT evaluation"
    }
}

# Data paths (modify as needed)
DATA_PATHS = {
    "stereoset": "./data/stereoset/test.json",
    "crows_gender": "./data/crows/gender.csv", 
    "crows_race": "./data/crows/race.csv",
    "crows_religion": "./data/crows/religion.csv",
    "seat": "./data/seat/"
}

# Output configurations
OUTPUT_CONFIGS = {
    "base_dir": "./bert_bias_results",
    "include_timestamp": True,
    "generate_plots": False,  # Set to True if you want visualization
    "save_individual_results": True,
    "save_comprehensive_results": True
}