#!/usr/bin/env python3
"""
Integrated Bias Mitigation Pipeline
Combines Unified Dataset Pipeline + Sycophancy Pipeline + Fairsteer Pipeline

This master pipeline orchestrates all three bias mitigation approaches:
1. Unified Dataset Evaluation (13 bias benchmarks)
2. Sycophancy-specific evaluation and mitigation
3. Fairsteer representation engineering and steering
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

class IntegratedPipelineRunner:
    """Orchestrates the complete integrated bias mitigation pipeline."""
    
    def __init__(self, base_dir: str = "/workspace/Algoverse"):
        self.base_dir = Path(base_dir)
        self.unified_dir = self.base_dir / "unified_pipeline"
        self.sycophancy_dir = self.base_dir / "sycophancy-interpretability"
        self.fairsteer_script = self.base_dir / "fairsteer_debiasing.py"
        
        self.results = {}
        self.start_time = datetime.now()
        
    def validate_environment(self) -> Dict[str, Any]:
        """Validate that all pipeline components are available."""
        validation_results = {
            "unified_pipeline": False,
            "sycophancy_pipeline": False,
            "fairsteer_pipeline": False,
            "datasets_available": False,
            "missing_components": []
        }
        
        # Check unified pipeline
        if (self.unified_dir / "run_unified_pipeline.py").exists():
            validation_results["unified_pipeline"] = True
        else:
            validation_results["missing_components"].append("Unified pipeline runner")
            
        # Check sycophancy pipeline
        if (self.sycophancy_dir / "evaluation" / "run_full_evaluation.sh").exists():
            validation_results["sycophancy_pipeline"] = True
        else:
            validation_results["missing_components"].append("Sycophancy evaluation scripts")
            
        # Check fairsteer pipeline  
        if self.fairsteer_script.exists():
            validation_results["fairsteer_pipeline"] = True
        else:
            validation_results["missing_components"].append("Fairsteer debiasing script")
            
        # Check dataset availability
        datasets_dir = self.base_dir / "datasets"
        required_datasets = ["crows-pairs", "bias-bench", "winobias", "winogender", 
                           "bbq", "bold", "biosbias", "truthfulqa"]
        available_datasets = [d for d in required_datasets 
                            if (datasets_dir / d).exists()]
        
        validation_results["datasets_available"] = len(available_datasets) >= 6
        validation_results["available_datasets"] = available_datasets
        validation_results["total_datasets"] = len(available_datasets)
        
        return validation_results
        
    def run_unified_evaluation(self, model_config: str, suite: str = "comprehensive") -> Dict[str, Any]:
        """Run the unified dataset evaluation pipeline."""
        print("\\n" + "="*60)
        print("RUNNING UNIFIED DATASET EVALUATION")
        print("="*60)
        
        cmd = [
            sys.executable, 
            str(self.unified_dir / "run_unified_pipeline.py"),
            "--model-config", model_config,
            "--suite", suite
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.unified_dir, 
                                  capture_output=True, text=True, timeout=1800)
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Unified evaluation timed out after 30 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unified evaluation failed: {str(e)}"
            }
            
    def run_sycophancy_evaluation(self, model_name: str) -> Dict[str, Any]:
        """Run the sycophancy-specific evaluation pipeline."""
        print("\\n" + "="*60)
        print("RUNNING SYCOPHANCY EVALUATION")
        print("="*60)
        
        # Check if evaluation script exists
        eval_script = self.sycophancy_dir / "evaluation" / "run_full_evaluation.sh"
        if not eval_script.exists():
            return {
                "success": False,
                "error": "Sycophancy evaluation script not found"
            }
            
        try:
            # Run sycophancy evaluation
            result = subprocess.run(
                ["bash", str(eval_script), model_name],
                cwd=self.sycophancy_dir / "evaluation",
                capture_output=True, text=True, timeout=2400
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "model_evaluated": model_name
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Sycophancy evaluation timed out after 40 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Sycophancy evaluation failed: {str(e)}"
            }
            
    def run_fairsteer_mitigation(self, model_config: str) -> Dict[str, Any]:
        """Run the Fairsteer bias mitigation."""
        print("\\n" + "="*60)
        print("RUNNING FAIRSTEER BIAS MITIGATION")
        print("="*60)
        
        if not self.fairsteer_script.exists():
            return {
                "success": False,
                "error": "Fairsteer script not found"
            }
            
        try:
            # Run fairsteer debiasing
            result = subprocess.run(
                [sys.executable, str(self.fairsteer_script), "--config", model_config],
                cwd=self.base_dir,
                capture_output=True, text=True, timeout=3600
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Fairsteer mitigation timed out after 60 minutes"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Fairsteer mitigation failed: {str(e)}"
            }
            
    def run_integrated_pipeline(self, model_config: str, model_name: str, 
                               suite: str = "comprehensive",
                               skip_sycophancy: bool = False,
                               skip_fairsteer: bool = False) -> Dict[str, Any]:
        """Run the complete integrated pipeline."""
        print("INTEGRATED BIAS MITIGATION PIPELINE")
        print("="*60)
        print(f"Model: {model_name}")
        print(f"Suite: {suite}")
        print(f"Start time: {self.start_time}")
        print("="*60)
        
        # Validate environment
        validation = self.validate_environment()
        self.results["validation"] = validation
        
        if validation["missing_components"]:
            print(f"⚠️  Missing components: {', '.join(validation['missing_components'])}")
            
        # Run unified evaluation
        if validation["unified_pipeline"] and validation["datasets_available"]:
            self.results["unified_evaluation"] = self.run_unified_evaluation(model_config, suite)
        else:
            self.results["unified_evaluation"] = {
                "success": False,
                "error": "Unified pipeline or datasets not available"
            }
            
        # Run sycophancy evaluation
        if not skip_sycophancy and validation["sycophancy_pipeline"]:
            self.results["sycophancy_evaluation"] = self.run_sycophancy_evaluation(model_name)
        else:
            self.results["sycophancy_evaluation"] = {
                "skipped": True,
                "reason": "Skipped by user" if skip_sycophancy else "Pipeline not available"
            }
            
        # Run fairsteer mitigation
        if not skip_fairsteer and validation["fairsteer_pipeline"]:
            self.results["fairsteer_mitigation"] = self.run_fairsteer_mitigation(model_config)
        else:
            self.results["fairsteer_mitigation"] = {
                "skipped": True,
                "reason": "Skipped by user" if skip_fairsteer else "Pipeline not available"
            }
            
        # Generate summary
        self.results["summary"] = self._generate_summary()
        
        return self.results
        
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of pipeline results."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        summary = {
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_seconds": duration,
            "components_run": [],
            "components_successful": [],
            "components_failed": []
        }
        
        # Check each component
        components = ["unified_evaluation", "sycophancy_evaluation", "fairsteer_mitigation"]
        
        for component in components:
            if component in self.results:
                result = self.results[component]
                if result.get("skipped"):
                    continue
                    
                summary["components_run"].append(component)
                if result.get("success"):
                    summary["components_successful"].append(component)
                else:
                    summary["components_failed"].append(component)
                    
        return summary
        
    def save_results(self, output_file: str):
        """Save pipeline results to file."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"\\n📊 Results saved to: {output_file}")


def main():
    """Main entry point for integrated pipeline."""
    parser = argparse.ArgumentParser(
        description="Integrated Bias Mitigation Pipeline (Unified + Sycophancy + Fairsteer)"
    )
    parser.add_argument("--model-config", required=True,
                       help="Path to model configuration file")
    parser.add_argument("--model-name", required=True,
                       help="Name of model for sycophancy evaluation")
    parser.add_argument("--suite", default="comprehensive",
                       choices=["comprehensive", "bias_focused", "sycophancy_focused", 
                               "working_baseline", "high_priority", "quick_evaluation"],
                       help="Evaluation suite to run")
    parser.add_argument("--skip-sycophancy", action="store_true",
                       help="Skip sycophancy evaluation")
    parser.add_argument("--skip-fairsteer", action="store_true", 
                       help="Skip Fairsteer mitigation")
    parser.add_argument("--output", default="integrated_pipeline_results.json",
                       help="Output file for results")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate environment, don't run pipeline")
    
    args = parser.parse_args()
    
    # Initialize pipeline runner
    runner = IntegratedPipelineRunner()
    
    if args.validate_only:
        # Just validate and exit
        validation = runner.validate_environment()
        print("\\nEnvironment Validation Results:")
        print("="*40)
        print(f"✅ Unified Pipeline: {'Available' if validation['unified_pipeline'] else 'Missing'}")
        print(f"✅ Sycophancy Pipeline: {'Available' if validation['sycophancy_pipeline'] else 'Missing'}")
        print(f"✅ Fairsteer Pipeline: {'Available' if validation['fairsteer_pipeline'] else 'Missing'}")
        print(f"✅ Datasets: {validation['total_datasets']}/8 available")
        
        if validation["missing_components"]:
            print(f"\\n⚠️  Missing: {', '.join(validation['missing_components'])}")
            
        return
    
    # Run integrated pipeline
    try:
        results = runner.run_integrated_pipeline(
            model_config=args.model_config,
            model_name=args.model_name,
            suite=args.suite,
            skip_sycophancy=args.skip_sycophancy,
            skip_fairsteer=args.skip_fairsteer
        )
        
        # Save results
        runner.save_results(args.output)
        
        # Print summary
        summary = results["summary"]
        print("\\n" + "="*60)
        print("INTEGRATED PIPELINE SUMMARY")
        print("="*60)
        print(f"Duration: {summary['total_duration_seconds']:.1f}s")
        print(f"Components run: {len(summary['components_run'])}")
        print(f"Successful: {len(summary['components_successful'])}")
        print(f"Failed: {len(summary['components_failed'])}")
        
        if summary['components_successful']:
            print(f"\\n✅ Successful: {', '.join(summary['components_successful'])}")
        if summary['components_failed']:
            print(f"\\n❌ Failed: {', '.join(summary['components_failed'])}")
            
        print("="*60)
        
    except KeyboardInterrupt:
        print("\\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()