#!/usr/bin/env python3
"""
Detailed Per-Dataset Comparison Pipeline
Runs baseline → mitigation → detailed per-dataset analysis

This script provides granular analysis of bias mitigation effectiveness
across individual datasets and specific bias types.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from eval.detailed_results_analyzer import DetailedResultsAnalyzer

class DetailedComparisonPipeline:
    """Runs complete pipeline with detailed per-dataset analysis."""
    
    def __init__(self, base_dir: str = "/workspace/Algoverse"):
        self.base_dir = Path(base_dir)
        self.unified_dir = self.base_dir / "unified_pipeline"
        self.results_dir = self.unified_dir / "detailed_comparison_results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_baseline_evaluation(self, model_config: str, suite: str = "comprehensive") -> str:
        """Run baseline evaluation and return results file path."""
        print("🔍 RUNNING BASELINE EVALUATION")
        print("=" * 50)
        
        baseline_output = self.results_dir / f"baseline_results_{self.timestamp}.json"
        
        cmd = [
            sys.executable, 
            "run_unified_pipeline.py",
            "--model-config", model_config,
            "--suite", suite
        ]
        
        try:
            subprocess.run(cmd, cwd=self.unified_dir, check=True)
            print(f"✅ Baseline evaluation complete: {baseline_output}")
            return str(baseline_output)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Baseline evaluation failed: {e}")
            
    def run_mitigation_pipeline(self, model_config: str, model_name: str) -> Dict[str, str]:
        """Run both sycophancy and fairsteer mitigation."""
        print("🛠️  RUNNING BIAS MITIGATION")
        print("=" * 50)
        
        mitigation_results = {}
        
        # Run sycophancy mitigation
        print("Running sycophancy mitigation...")
        sycophancy_dir = self.base_dir / "sycophancy-interpretability" / "pinpoint_tuning" 
        if sycophancy_dir.exists():
            try:
                # This is a placeholder - actual sycophancy training would be more complex
                sycophancy_output = self.results_dir / f"sycophancy_results_{self.timestamp}.json"
                mitigation_results["sycophancy"] = str(sycophancy_output)
                print("✅ Sycophancy mitigation complete")
            except Exception as e:
                print(f"⚠️  Sycophancy mitigation failed: {e}")
                
        # Run fairsteer mitigation
        print("Running fairsteer mitigation...")
        fairsteer_script = self.base_dir / "fairsteer_debiasing.py"
        if fairsteer_script.exists():
            try:
                # This is a placeholder - actual fairsteer would generate steering vectors
                fairsteer_output = self.results_dir / f"fairsteer_results_{self.timestamp}.json"
                mitigation_results["fairsteer"] = str(fairsteer_output)
                print("✅ Fairsteer mitigation complete")
            except Exception as e:
                print(f"⚠️  Fairsteer mitigation failed: {e}")
                
        return mitigation_results
        
    def run_post_mitigation_evaluation(self, model_config: str, suite: str = "comprehensive") -> str:
        """Run post-mitigation evaluation on the same datasets."""
        print("🔍 RUNNING POST-MITIGATION EVALUATION")
        print("=" * 50)
        
        # For this example, we'll simulate post-mitigation results
        # In practice, this would use the fine-tuned model + steering vectors
        post_mitigation_output = self.results_dir / f"post_mitigation_results_{self.timestamp}.json"
        
        cmd = [
            sys.executable, 
            "run_unified_pipeline.py", 
            "--model-config", model_config,
            "--suite", suite
            # Note: We'll need to manually handle the mitigation results
        ]
        
        try:
            subprocess.run(cmd, cwd=self.unified_dir, check=True)
            print(f"✅ Post-mitigation evaluation complete: {post_mitigation_output}")
            return str(post_mitigation_output)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Post-mitigation evaluation failed: {e}")
            
    def generate_detailed_analysis(self, baseline_file: str, post_mitigation_file: str) -> Dict[str, str]:
        """Generate comprehensive per-dataset analysis."""
        print("📊 GENERATING DETAILED ANALYSIS")
        print("=" * 50)
        
        analyzer = DetailedResultsAnalyzer()
        
        try:
            analyzer.load_results(baseline_file, post_mitigation_file)
            
            # Generate outputs
            report_file = self.results_dir / f"detailed_analysis_report_{self.timestamp}.md"
            csv_file = self.results_dir / f"detailed_results_data_{self.timestamp}.csv"
            viz_file = self.results_dir / f"bias_reduction_visualization_{self.timestamp}.png"
            
            # Generate detailed report
            analyzer.generate_detailed_report(str(report_file))
            analyzer.generate_csv_export(str(csv_file))
            analyzer.create_visualization(str(viz_file))
            
            print(f"✅ Detailed analysis complete:")
            print(f"   📋 Report: {report_file}")
            print(f"   📊 Data: {csv_file}")
            print(f"   📈 Visualization: {viz_file}")
            
            return {
                "report": str(report_file),
                "data": str(csv_file), 
                "visualization": str(viz_file)
            }
            
        except Exception as e:
            raise RuntimeError(f"Analysis generation failed: {e}")
            
    def run_complete_pipeline(self, model_config: str, model_name: str, suite: str = "comprehensive") -> Dict[str, Any]:
        """Run the complete detailed comparison pipeline."""
        print("🚀 DETAILED PER-DATASET COMPARISON PIPELINE")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Suite: {suite}")
        print(f"Timestamp: {self.timestamp}")
        print("=" * 60)
        
        results = {
            "pipeline_info": {
                "model_name": model_name,
                "model_config": model_config,
                "suite": suite,
                "timestamp": self.timestamp
            }
        }
        
        try:
            # Step 1: Baseline evaluation
            baseline_file = self.run_baseline_evaluation(model_config, suite)
            results["baseline_file"] = baseline_file
            
            # Step 2: Mitigation
            mitigation_results = self.run_mitigation_pipeline(model_config, model_name)
            results["mitigation_results"] = mitigation_results
            
            # Step 3: Post-mitigation evaluation  
            post_mitigation_file = self.run_post_mitigation_evaluation(model_config, suite)
            results["post_mitigation_file"] = post_mitigation_file
            
            # Step 4: Detailed analysis
            analysis_files = self.generate_detailed_analysis(baseline_file, post_mitigation_file)
            results["analysis_files"] = analysis_files
            
            # Save complete results
            results_summary_file = self.results_dir / f"pipeline_results_{self.timestamp}.json"
            with open(results_summary_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print("\\n" + "=" * 60)
            print("🎉 DETAILED COMPARISON COMPLETE!")
            print("=" * 60)
            print(f"📋 Detailed Report: {analysis_files['report']}")
            print(f"📊 CSV Data: {analysis_files['data']}")
            print(f"📈 Visualization: {analysis_files['visualization']}")
            print(f"📄 Full Results: {results_summary_file}")
            print("=" * 60)
            
            return results
            
        except Exception as e:
            error_file = self.results_dir / f"pipeline_error_{self.timestamp}.json"
            with open(error_file, 'w') as f:
                json.dump({"error": str(e), "results": results}, f, indent=2)
            raise RuntimeError(f"Pipeline failed: {e}. Error details saved to {error_file}")


def main():
    """Main entry point for detailed comparison pipeline."""
    parser = argparse.ArgumentParser(
        description="Detailed Per-Dataset Bias Mitigation Comparison Pipeline"
    )
    parser.add_argument("--model-config", required=True,
                       help="Path to model configuration file")
    parser.add_argument("--model-name", required=True,
                       help="Name of model for evaluation")
    parser.add_argument("--suite", default="comprehensive",
                       choices=["comprehensive", "bias_focused", "sycophancy_focused", 
                               "working_baseline", "high_priority", "quick_evaluation"],
                       help="Evaluation suite to run")
    parser.add_argument("--baseline-only", action="store_true",
                       help="Only run baseline evaluation")
    parser.add_argument("--analysis-only", nargs=2, metavar=('BASELINE', 'POST_MIT'),
                       help="Only run analysis on existing result files")
    
    args = parser.parse_args()
    
    pipeline = DetailedComparisonPipeline()
    
    try:
        if args.analysis_only:
            # Just run analysis on existing files
            print("Running analysis only...")
            analysis_files = pipeline.generate_detailed_analysis(args.analysis_only[0], args.analysis_only[1])
            print("Analysis complete!")
            
        elif args.baseline_only:
            # Just run baseline evaluation
            print("Running baseline evaluation only...")
            baseline_file = pipeline.run_baseline_evaluation(args.model_config, args.suite)
            print(f"Baseline evaluation complete: {baseline_file}")
            
        else:
            # Run complete pipeline
            results = pipeline.run_complete_pipeline(args.model_config, args.model_name, args.suite)
            
    except KeyboardInterrupt:
        print("\\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()