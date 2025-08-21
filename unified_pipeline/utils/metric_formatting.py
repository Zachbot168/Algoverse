#!/usr/bin/env python3
"""
Utility functions for formatting metrics with appropriate precision
"""

def format_metric_with_precision(value, dataset_name, metric_name):
    """
    Format metrics with appropriate precision to show meaningful differences.
    
    For very small values (like BOLD sentiment bias), show more decimal places.
    For larger values, use standard 3-4 decimal formatting.
    """
    if value == 0.0:
        return "0.0"
    
    # Special handling for known small-value metrics
    if (dataset_name == "BOLD" and "sentiment" in metric_name.lower()) or \
       (dataset_name == "SEAT" and "effect" in metric_name.lower()):
        # For very small values, show enough precision to see differences
        if abs(value) < 0.001:
            return f"{value:.6f}"  # 6 decimal places for sub-millionth values
        elif abs(value) < 0.01:
            return f"{value:.5f}"  # 5 decimal places for sub-centesimal values
        elif abs(value) < 0.1:
            return f"{value:.4f}"  # 4 decimal places for sub-decimal values
        else:
            return f"{value:.3f}"  # 3 decimal places for normal values
    
    # Standard formatting for other metrics
    return f"{value:.3f}"

def get_dataset_primary_metric_name(dataset_name):
    """
    Get the primary metric name for each dataset for proper formatting.
    """
    metric_mapping = {
        "CrowsPairs": "crows_pairs_bias_score",
        "StereoSet": "stereoset_bias_score", 
        "WinoBias": "winobias_accuracy",
        "WinoGender": "winogender_accuracy",
        "BBQ": "bbq_accuracy",
        "SEAT": "seat_avg_effect_size",
        "BOLD": "bold_sentiment_bias",
        "BiosBias": "biosbias_bias_score",
        "TruthfulQA": "truthfulqa_truthful_pct",
        "SycophancyEval": "sycophancy_eval_non_sycophantic_pct"
    }
    return metric_mapping.get(dataset_name, f"{dataset_name.lower()}_score")

def format_dataset_score(value, dataset_name):
    """
    Format a dataset score with appropriate precision and context.
    """
    metric_name = get_dataset_primary_metric_name(dataset_name)
    formatted_value = format_metric_with_precision(value, dataset_name, metric_name)
    
    # Add interpretation context
    if dataset_name == "BOLD" and "sentiment" in metric_name:
        return f"{formatted_value} (sentiment bias)"
    elif dataset_name == "SEAT" and "effect" in metric_name:
        return f"{formatted_value} (avg effect size)"
    else:
        return formatted_value