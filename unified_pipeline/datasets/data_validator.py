#!/usr/bin/env python3
"""
Dataset Validation System

Validates that all required datasets are present and contain real data
before allowing evaluation to proceed. Prevents silent failures that
could lead to fake results.
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

class DatasetValidator:
    """Validates dataset integrity and availability."""
    
    def __init__(self, datasets_dir: str = "../datasets"):
        """Initialize validator with datasets directory."""
        self.datasets_dir = Path(datasets_dir)
        self.logger = logging.getLogger(__name__)
        
        # Expected dataset structure
        self.dataset_specs = {
            "crows-pairs": {
                "required_files": ["data/crows_pairs_anonymized.csv"],
                "description": "CrowS-Pairs stereotype evaluation dataset",
                "min_samples": 1000,
                "validation_func": self._validate_crows_pairs
            },
            "winobias": {
                "required_files": [
                    "wino/data/anti_stereotyped_type1.txt.dev",
                    "wino/data/pro_stereotyped_type1.txt.dev"
                ],
                "description": "WinoBias gender coreference dataset", 
                "min_samples": 100,
                "validation_func": self._validate_winobias
            },
            "winogender": {
                "required_files": ["data/templates.tsv"],
                "description": "WinoGender coreference resolution dataset",
                "min_samples": 50,
                "validation_func": self._validate_winogender
            },
            "bbq": {
                "required_files": ["BBQ.csv"],
                "description": "BBQ bias benchmark for QA",
                "min_samples": 1000,
                "validation_func": self._validate_bbq
            },
            "stereoset": {
                "required_files": ["data/bias-bench/stereoset/test.json"],
                "description": "StereoSet stereotype evaluation",
                "min_samples": 1000,
                "validation_func": self._validate_stereoset
            },
            "bold": {
                "required_files": ["prompts.json"],
                "description": "BOLD demographic fairness dataset",
                "min_samples": 20,
                "validation_func": self._validate_bold
            },
            "truthfulqa": {
                "required_files": ["TruthfulQA.csv"],
                "description": "TruthfulQA truthfulness evaluation",
                "min_samples": 100,
                "validation_func": self._validate_truthfulqa
            }
        }
    
    def validate_all_datasets(self) -> Dict[str, Dict]:
        """
        Validate all datasets.
        
        Returns:
            Dictionary with validation results for each dataset
        """
        results = {}
        
        print("🔍 Validating dataset integrity...")
        print("=" * 50)
        
        for dataset_name, spec in self.dataset_specs.items():
            print(f"\n📊 Validating {dataset_name}...")
            result = self.validate_dataset(dataset_name)
            results[dataset_name] = result
            
            status = "✅ VALID" if result["valid"] else "❌ INVALID"
            print(f"   {status}: {result['message']}")
            
            if result["valid"] and "sample_count" in result:
                print(f"   📈 Samples: {result['sample_count']}")
        
        # Summary
        valid_count = sum(1 for r in results.values() if r["valid"])
        total_count = len(results)
        
        print(f"\n📋 VALIDATION SUMMARY")
        print(f"   Valid datasets: {valid_count}/{total_count}")
        
        if valid_count == 0:
            print("❌ No datasets available - cannot run evaluation")
        elif valid_count < total_count // 2:
            print("⚠️  Limited datasets available - evaluation may be incomplete")
        else:
            print("✅ Sufficient datasets available for evaluation")
        
        return results
    
    def validate_dataset(self, dataset_name: str) -> Dict:
        """
        Validate a single dataset.
        
        Args:
            dataset_name: Name of dataset to validate
            
        Returns:
            Validation result dictionary
        """
        if dataset_name not in self.dataset_specs:
            return {
                "valid": False,
                "message": f"Unknown dataset: {dataset_name}",
                "error": "unknown_dataset"
            }
        
        spec = self.dataset_specs[dataset_name]
        dataset_path = self.datasets_dir / dataset_name
        
        # Check if dataset directory exists
        if not dataset_path.exists():
            return {
                "valid": False,
                "message": f"Dataset directory not found: {dataset_path}",
                "error": "missing_directory"
            }
        
        # Check required files
        missing_files = []
        for required_file in spec["required_files"]:
            file_path = dataset_path / required_file
            if not file_path.exists():
                missing_files.append(required_file)
        
        if missing_files:
            return {
                "valid": False,
                "message": f"Missing required files: {missing_files}",
                "error": "missing_files",
                "missing_files": missing_files
            }
        
        # Run dataset-specific validation
        try:
            validation_result = spec["validation_func"](dataset_path)
            return validation_result
        except Exception as e:
            return {
                "valid": False,
                "message": f"Validation error: {str(e)}",
                "error": "validation_exception"
            }
    
    def _validate_crows_pairs(self, dataset_path: Path) -> Dict:
        """Validate CrowS-Pairs dataset."""
        csv_path = dataset_path / "data/crows_pairs_anonymized.csv"
        
        try:
            df = pd.read_csv(csv_path)
            
            # Check required columns
            required_cols = ["sent_more", "sent_less", "stereo_antistereo", "bias_type"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return {
                    "valid": False,
                    "message": f"Missing columns: {missing_cols}",
                    "error": "missing_columns"
                }
            
            sample_count = len(df)
            if sample_count < 1000:
                return {
                    "valid": False,
                    "message": f"Too few samples: {sample_count} (expected ≥1000)",
                    "error": "insufficient_samples"
                }
            
            return {
                "valid": True,
                "message": "CrowS-Pairs dataset valid",
                "sample_count": sample_count,
                "bias_types": df["bias_type"].unique().tolist()
            }
            
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error reading CSV: {str(e)}",
                "error": "read_error"
            }
    
    def _validate_winobias(self, dataset_path: Path) -> Dict:
        """Validate WinoBias dataset."""
        try:
            sample_count = 0
            file_counts = {}
            
            for file_name in ["anti_stereotyped_type1.txt.dev", "pro_stereotyped_type1.txt.dev"]:
                file_path = dataset_path / "wino/data" / file_name
                
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    count = len([line for line in lines if line.strip()])
                    file_counts[file_name] = count
                    sample_count += count
            
            if sample_count < 100:
                return {
                    "valid": False,
                    "message": f"Too few samples: {sample_count} (expected ≥100)",
                    "error": "insufficient_samples"
                }
            
            return {
                "valid": True,
                "message": "WinoBias dataset valid",
                "sample_count": sample_count,
                "file_breakdown": file_counts
            }
            
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error reading files: {str(e)}",
                "error": "read_error"
            }
    
    def _validate_winogender(self, dataset_path: Path) -> Dict:
        """Validate WinoGender dataset."""
        try:
            tsv_path = dataset_path / "data/templates.tsv"
            df = pd.read_csv(tsv_path, sep='\t')
            
            sample_count = len(df)
            if sample_count < 50:
                return {
                    "valid": False,
                    "message": f"Too few samples: {sample_count} (expected ≥50)",
                    "error": "insufficient_samples"
                }
            
            return {
                "valid": True,
                "message": "WinoGender dataset valid",
                "sample_count": sample_count
            }
            
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error reading TSV: {str(e)}",
                "error": "read_error"
            }
    
    def _validate_bbq(self, dataset_path: Path) -> Dict:
        """Validate BBQ dataset."""
        try:
            csv_path = dataset_path / "BBQ.csv"
            df = pd.read_csv(csv_path)
            
            required_cols = ["context", "question", "ans0", "ans1", "ans2"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return {
                    "valid": False,
                    "message": f"Missing columns: {missing_cols}",
                    "error": "missing_columns"
                }
            
            sample_count = len(df)
            if sample_count < 1000:
                return {
                    "valid": False,
                    "message": f"Too few samples: {sample_count} (expected ≥1000)",
                    "error": "insufficient_samples"
                }
            
            return {
                "valid": True,
                "message": "BBQ dataset valid",
                "sample_count": sample_count
            }
            
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error reading CSV: {str(e)}",
                "error": "read_error"
            }
    
    def _validate_stereoset(self, dataset_path: Path) -> Dict:
        """Validate StereoSet dataset."""
        try:
            json_path = dataset_path / "data/bias-bench/stereoset/test.json"
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if "data" not in data:
                return {
                    "valid": False,
                    "message": "Missing 'data' key in JSON",
                    "error": "invalid_format"
                }
            
            sample_count = len(data["data"]["intrasentence"]) + len(data["data"]["intersentence"])
            
            if sample_count < 1000:
                return {
                    "valid": False,
                    "message": f"Too few samples: {sample_count} (expected ≥1000)",
                    "error": "insufficient_samples"
                }
            
            return {
                "valid": True,
                "message": "StereoSet dataset valid",
                "sample_count": sample_count
            }
            
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error reading JSON: {str(e)}",
                "error": "read_error"
            }
    
    def _validate_bold(self, dataset_path: Path) -> Dict:
        """Validate BOLD dataset."""
        try:
            json_path = dataset_path / "prompts.json"
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                sample_count = len(data)
            elif isinstance(data, dict):
                sample_count = sum(len(v) for v in data.values() if isinstance(v, list))
            else:
                return {
                    "valid": False,
                    "message": "Invalid JSON format",
                    "error": "invalid_format"
                }
            
            if sample_count < 20:
                return {
                    "valid": False,
                    "message": f"Too few samples: {sample_count} (expected ≥20)",
                    "error": "insufficient_samples"
                }
            
            return {
                "valid": True,
                "message": "BOLD dataset valid",
                "sample_count": sample_count
            }
            
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error reading JSON: {str(e)}",
                "error": "read_error"
            }
    
    def _validate_truthfulqa(self, dataset_path: Path) -> Dict:
        """Validate TruthfulQA dataset."""
        try:
            csv_path = dataset_path / "TruthfulQA.csv"
            df = pd.read_csv(csv_path)
            
            required_cols = ["Question", "Best Answer", "Correct Answers", "Incorrect Answers"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return {
                    "valid": False,
                    "message": f"Missing columns: {missing_cols}",
                    "error": "missing_columns"
                }
            
            sample_count = len(df)
            if sample_count < 100:
                return {
                    "valid": False,
                    "message": f"Too few samples: {sample_count} (expected ≥100)",
                    "error": "insufficient_samples"
                }
            
            return {
                "valid": True,
                "message": "TruthfulQA dataset valid",
                "sample_count": sample_count
            }
            
        except Exception as e:
            return {
                "valid": False,
                "message": f"Error reading CSV: {str(e)}",
                "error": "read_error"
            }
    
    def get_available_datasets(self) -> List[str]:
        """Get list of valid datasets."""
        results = self.validate_all_datasets()
        return [name for name, result in results.items() if result["valid"]]
    
    def require_datasets(self, required_datasets: List[str]) -> bool:
        """
        Ensure required datasets are available.
        
        Args:
            required_datasets: List of dataset names that must be available
            
        Returns:
            True if all required datasets are valid, False otherwise
        """
        results = self.validate_all_datasets()
        
        missing_datasets = []
        for dataset in required_datasets:
            if dataset not in results or not results[dataset]["valid"]:
                missing_datasets.append(dataset)
        
        if missing_datasets:
            print(f"❌ Required datasets missing: {missing_datasets}")
            print("Please run: ./enhanced_pull_datasets.sh")
            return False
        
        print(f"✅ All required datasets available: {required_datasets}")
        return True

def main():
    """CLI entry point for dataset validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Algoverse datasets")
    parser.add_argument("--datasets-dir", default="../datasets", help="Path to datasets directory")
    parser.add_argument("--require", nargs="+", help="Datasets that must be available")
    
    args = parser.parse_args()
    
    validator = DatasetValidator(args.datasets_dir)
    
    if args.require:
        success = validator.require_datasets(args.require)
        exit(0 if success else 1)
    else:
        results = validator.validate_all_datasets()
        valid_count = sum(1 for r in results.values() if r["valid"])
        exit(0 if valid_count > 0 else 1)

if __name__ == "__main__":
    main()