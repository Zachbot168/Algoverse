"""
Script to process all StereoSet bias evaluation results from bias-bench
"""

import json
import os
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Any
import statistics

def load_json_file(filepath: Path) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}

def extract_model_name(filename: str) -> str:
    """Extract model name from filename."""
    # Remove .json extension and any common prefixes/suffixes
    name = filename.replace('.json', '')
    # Handle common naming patterns
    name = name.replace('_stereoset', '').replace('stereoset_', '')
    return name

def process_stereoset_results(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process StereoSet results and extract key metrics."""
    results = {}
    
    # Extract overall scores if available
    if 'overall' in data:
        overall = data['overall']
        results.update({
            'overall_bias_score': overall.get('bias_score'),
            'overall_stereotype_score': overall.get('stereotype_score'),
            'overall_language_modeling_score': overall.get('language_modeling_score'),
            'overall_icat_score': overall.get('icat_score')
        })
    
    # Extract intrasentence results
    if 'intrasentence' in data:
        intra = data['intrasentence']
        results.update({
            'intra_bias_score': intra.get('bias_score'),
            'intra_stereotype_score': intra.get('stereotype_score'),
            'intra_language_modeling_score': intra.get('language_modeling_score'),
            'intra_icat_score': intra.get('icat_score')
        })
        
        # Extract domain-specific results
        if 'domains' in intra:
            for domain, domain_data in intra['domains'].items():
                prefix = f'intra_{domain}'
                results.update({
                    f'{prefix}_bias_score': domain_data.get('bias_score'),
                    f'{prefix}_stereotype_score': domain_data.get('stereotype_score'),
                    f'{prefix}_language_modeling_score': domain_data.get('language_modeling_score'),
                    f'{prefix}_icat_score': domain_data.get('icat_score')
                })
    
    # Extract intersentence results if available
    if 'intersentence' in data:
        inter = data['intersentence']
        results.update({
            'inter_bias_score': inter.get('bias_score'),
            'inter_stereotype_score': inter.get('stereotype_score'),
            'inter_language_modeling_score': inter.get('language_modeling_score'),
            'inter_icat_score': inter.get('icat_score')
        })
    
    return results

def analyze_results(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the collected results and compute summary statistics."""
    analysis = {}
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Overall statistics
    analysis['summary_stats'] = {}
    for col in numeric_cols:
        if col in df.columns and not df[col].isna().all():
            analysis['summary_stats'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
    
    # Best and worst performing models for key metrics
    key_metrics = ['overall_bias_score', 'overall_icat_score', 'intra_bias_score', 'intra_icat_score']
    analysis['rankings'] = {}
    
    for metric in key_metrics:
        if metric in df.columns and not df[metric].isna().all():
            # For bias scores, lower is better; for ICAT scores, higher is better
            if 'bias_score' in metric:
                best_idx = df[metric].idxmin()
                worst_idx = df[metric].idxmax()
                analysis['rankings'][metric] = {
                    'best_model': df.loc[best_idx, 'model_name'],
                    'best_score': df.loc[best_idx, metric],
                    'worst_model': df.loc[worst_idx, 'model_name'],
                    'worst_score': df.loc[worst_idx, metric]
                }
            elif 'icat_score' in metric:
                best_idx = df[metric].idxmax()
                worst_idx = df[metric].idxmin()
                analysis['rankings'][metric] = {
                    'best_model': df.loc[best_idx, 'model_name'],
                    'best_score': df.loc[best_idx, metric],
                    'worst_model': df.loc[worst_idx, 'model_name'],
                    'worst_score': df.loc[worst_idx, metric]
                }
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description='Process StereoSet bias evaluation results')
    parser.add_argument('--input-dir', default='bias-bench\quick_bias_results\stereoset',
                        help='Directory containing StereoSet result JSON files')
    parser.add_argument('--output-dir', default='processed_results',
                        help='Directory to save processed results')
    parser.add_argument('--output-format', choices=['csv', 'json', 'both'], default='both',
                        help='Output format for results')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Find all JSON files
    json_files = list(input_dir.glob('*.json'))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    all_results = []
    
    for json_file in json_files:
        print(f"Processing {json_file.name}...")
        
        data = load_json_file(json_file)
        if not data:
            continue
        
        model_name = extract_model_name(json_file.name)
        results = process_stereoset_results(data)
        results['model_name'] = model_name
        results['filename'] = json_file.name
        
        all_results.append(results)
    
    if not all_results:
        print("No valid results found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns to put model info first
    cols = ['model_name', 'filename'] + [col for col in df.columns if col not in ['model_name', 'filename']]
    df = df[cols]
    
    print(f"\nProcessed {len(df)} models successfully")
    print(f"Columns available: {list(df.columns)}")
    
    # Save results
    if args.output_format in ['csv', 'both']:
        csv_path = output_dir / 'stereoset_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    
    if args.output_format in ['json', 'both']:
        json_path = output_dir / 'stereoset_results.json'
        df.to_json(json_path, orient='records', indent=2)
        print(f"Results saved to {json_path}")
    
    # Perform analysis
    print("\nPerforming analysis...")
    analysis = analyze_results(df)
    
    # Save analysis
    analysis_path = output_dir / 'stereoset_analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"Analysis saved to {analysis_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    # Print key metrics summary
    key_metrics = ['overall_bias_score', 'overall_icat_score']
    for metric in key_metrics:
        if metric in analysis['summary_stats']:
            stats = analysis['summary_stats'][metric]
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}")
    
    # Print best/worst models
    print(f"\nMODEL RANKINGS:")
    for metric, ranking in analysis['rankings'].items():
        print(f"\n{metric}:")
        print(f"  Best:  {ranking['best_model']} ({ranking['best_score']:.4f})")
        print(f"  Worst: {ranking['worst_model']} ({ranking['worst_score']:.4f})")

if __name__ == "__main__":
    main()