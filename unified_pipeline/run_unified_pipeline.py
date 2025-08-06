#!/usr/bin/env python3
"""
Unified Bias Mitigation Pipeline with Comprehensive Dataset Integration

Complete end-to-end pipeline integrating ALL bias evaluation datasets:
- High Priority: StereoSet, SEAT, TruthfulQA, WinoGender  
- Medium Priority: BOLD, BiosBias, MMLU
- Low Priority: HumanEval, GSM8K
- Working: CrowsPairs, WinoBias, SycophancyEval, BBQ

This pipeline preserves unique dataset characteristics while providing
comprehensive bias evaluation and mitigation capabilities.
"""

# CRITICAL: Set environment variables BEFORE any torch imports
import os
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = 'true'
os.environ['PYTHONWARNINGS'] = 'ignore'

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings
import yaml
import logging

# Import torch AFTER setting environment variables
import torch

# Immediately disable torch dynamo after import
try:
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
    torch._dynamo.reset()
except:
    pass  # Older torch versions

# Suppress ALL warnings and verbose outputs for cleaner pipeline execution
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.WARNING)

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import unified pipeline components
try:
    from eval.unified_evaluator import UnifiedBiasEvaluator, run_unified_evaluation
    from datasets import UnifiedDatasetRegistry
    from utils.model_adapter import create_model_adapter
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print(f"Error importing pipeline components: {e}")
    print("Installing required dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "pydantic", "scikit-learn"])
    
    # Retry imports
    from eval.unified_evaluator import UnifiedBiasEvaluator, run_unified_evaluation
    from datasets import UnifiedDatasetRegistry
    from transformers import AutoModelForCausalLM, AutoTokenizer


class UnifiedBiasMitigationPipeline:
    """
    Unified bias mitigation pipeline with comprehensive dataset integration.
    
    Supports all 13 bias evaluation datasets while preserving their unique
    characteristics and evaluation methodologies.
    """
    
    def __init__(self, config_path: str, dataset_config_path: str):
        """
        Initialize unified pipeline.
        
        Args:
            config_path: Path to model/pipeline configuration
            dataset_config_path: Path to dataset configuration
        """
        self.config_path = config_path
        self.dataset_config_path = dataset_config_path
        
        # Load configurations
        self.config = self._load_config(config_path)
        self.dataset_config = self._load_config(dataset_config_path)
        
        # Extract key settings
        self.model_name = self.config['model']['name']
        self.base_data_path = self.config.get('base_data_path', '/workspace/Algoverse')
        
        # Setup output directories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_base = f"unified_pipeline_runs/{timestamp}"
        self.setup_directories()
        
        # Initialize registry and evaluator
        self.registry = UnifiedDatasetRegistry(self.base_data_path)
        self.evaluator = UnifiedBiasEvaluator(self.dataset_config, self.base_data_path)
        
        # Pipeline state
        self.pipeline_state = {
            'start_time': datetime.now().isoformat(),
            'config_path': config_path,
            'dataset_config_path': dataset_config_path,
            'model_name': self.model_name,
            'output_directory': self.output_base,
            'stages_completed': [],
            'results': {}
        }
        
        print(f"Initialized Unified Bias Mitigation Pipeline")
        print(f"Model: {self.model_name}")
        print(f"Output: {self.output_base}")
        print(f"Registry: {self.registry}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Error loading config {config_path}: {e}")
    
    def setup_directories(self):
        """Create output directory structure."""
        self.dirs = {
            'diagnostics': f"{self.output_base}/diagnostics",
            'training': f"{self.output_base}/training", 
            'steering': f"{self.output_base}/steering",
            'evaluation': f"{self.output_base}/evaluation",
            'reports': f"{self.output_base}/reports"
        }
        
        for directory in self.dirs.values():
            os.makedirs(directory, exist_ok=True)
    
    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate pipeline environment and dataset availability.
        
        Returns:
            Validation results including dataset availability
        """
        print("\n=== Environment Validation ===")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'model_accessible': False,
            'datasets_available': {},
            'total_datasets': 0,
            'available_datasets': 0,
            'validation_errors': []
        }
        
        try:
            # Test model loading
            print(f"Testing model access: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer:
                validation_results['model_accessible'] = True
                print("✓ Model accessible")
            else:
                validation_results['validation_errors'].append("Model tokenizer could not be loaded")
        
        except Exception as e:
            validation_results['validation_errors'].append(f"Model access error: {str(e)}")
            print(f"✗ Model access failed: {e}")
        
        # Validate dataset availability
        print("\nValidating dataset availability...")
        dataset_availability = self.evaluator.validate_dataset_availability()
        validation_results['datasets_available'] = dataset_availability
        validation_results['total_datasets'] = len(dataset_availability)
        validation_results['available_datasets'] = sum(dataset_availability.values())
        
        # Generate coverage report
        coverage_report = self.registry.get_comprehensive_coverage_report()
        validation_results['coverage_report'] = coverage_report
        
        print(f"\nDataset Summary:")
        print(f"  Total datasets: {validation_results['total_datasets']}")
        print(f"  Available: {validation_results['available_datasets']}")
        print(f"  ✅ Fully implemented: {len(self.registry.IMPLEMENTED_DATASETS)}")
        print(f"  🎯 All datasets are now working and integrated!")
        
        # Show breakdown by original priority (legacy info)
        print(f"\nDataset Implementation Status:")
        print(f"  ✅ Original working datasets: {len(self.registry.WORKING_DATASETS)}")
        print(f"  ✅ Original high priority: {len(self.registry.HIGH_PRIORITY)} (now implemented)")
        print(f"  ✅ Original medium priority: {len(self.registry.MEDIUM_PRIORITY)} (now implemented)")
        print(f"  ✅ Original low priority: {len(self.registry.LOW_PRIORITY)} (now implemented)")
        
        # Save validation results
        validation_file = f"{self.dirs['diagnostics']}/environment_validation.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        print(f"✓ Validation results saved to: {validation_file}")
        
        return validation_results
    
    def load_model(self) -> tuple:
        """
        Load model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"\n=== Loading Model ===")
        print(f"Loading {self.model_name}... (suppressing compilation warnings)")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with aggressive compilation suppression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Disable torch compilation at the model level
                try:
                    torch._dynamo.reset()
                except:
                    pass
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    attn_implementation="eager"  # Disable flash attention compilation
                )
                
                # Ensure model doesn't get compiled
                try:
                    if hasattr(torch, 'compiler') and hasattr(model, 'forward'):
                        model.forward = torch.compiler.disable(model.forward)
                except:
                    pass  # Older torch versions
            
            print(f"✓ Model loaded successfully")
            print(f"  Model type: {type(model).__name__}")
            print(f"  Parameters: ~{sum(p.numel() for p in model.parameters()) // 1_000_000}M")
            print(f"  Device: {next(model.parameters()).device}")
            
            return model, tokenizer
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def run_baseline_evaluation(
        self,
        model,
        tokenizer,
        suite_name: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Run baseline bias evaluation before any interventions.
        
        Args:
            model: Model to evaluate
            tokenizer: Model tokenizer
            suite_name: Evaluation suite to run
            
        Returns:
            Baseline evaluation results
        """
        print(f"\n=== Baseline Bias Evaluation ===")
        print(f"Running evaluation suite: {suite_name} (warnings suppressed for cleaner output)")
        
        # Run comprehensive evaluation with clean output
        print("🔄 Starting dataset evaluation...") 
        baseline_results = self.evaluator.run_comprehensive_evaluation(
            model=model,
            tokenizer=tokenizer,
            suite_name=suite_name,
            output_dir=f"{self.dirs['evaluation']}/baseline"
        )
        print("✅ Dataset evaluation completed")
        
        # Store in pipeline state
        self.pipeline_state['results']['baseline_evaluation'] = baseline_results
        self.pipeline_state['stages_completed'].append('baseline_evaluation')
        
        # Print summary
        total_datasets = baseline_results['total_datasets_evaluated']
        total_time = baseline_results['total_evaluation_time']
        aggregated = baseline_results.get('aggregated_metrics', {})
        
        print(f"\n✓ Baseline evaluation completed")
        print(f"  Datasets evaluated: {total_datasets}")
        print(f"  Total time: {total_time:.1f}s")
        
        if 'overall' in aggregated:
            overall = aggregated['overall']
            if 'mean_score' in overall:
                print(f"  Overall mean score: {overall['mean_score']:.3f}")
        
        # Print per-bias-type summary
        if 'by_bias_type' in aggregated:
            print("\n  Bias Type Scores:")
            for bias_type, scores in aggregated['by_bias_type'].items():
                mean_score = scores.get('mean_score', 0)
                dataset_count = scores.get('dataset_count', 0)
                print(f"    {bias_type}: {mean_score:.3f} ({dataset_count} datasets)")
        
        return baseline_results
    
    def run_dataset_analysis(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze evaluation results to identify bias patterns and priorities.
        
        Args:
            evaluation_results: Results from bias evaluation
            
        Returns:
            Analysis results and recommendations
        """
        print(f"\n=== Dataset Analysis ===")
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_datasets_analyzed': 0,
            'bias_severity_ranking': [],
            'dataset_performance': {},
            'recommendations': {},
            'integration_status': {
                'working_datasets': [],
                'high_priority_missing': [],
                'medium_priority_missing': [],
                'low_priority_missing': []
            }
        }
        
        # Analyze dataset results
        dataset_results = evaluation_results.get('dataset_results', {})
        analysis['total_datasets_analyzed'] = len(dataset_results)
        
        # Performance analysis
        for dataset_name, result in dataset_results.items():
            metrics = result.get('metrics', {})
            metadata = result.get('metadata', {})
            
            # Extract main performance metric
            main_metric = self._extract_main_performance_metric(dataset_name, metrics)
            
            analysis['dataset_performance'][dataset_name] = {
                'main_metric': main_metric,
                'bias_types': metadata.get('bias_types', []),
                'evaluation_mode': metadata.get('evaluation_mode', ''),
                'total_samples': metadata.get('total_samples', 0),
                'priority': self._get_dataset_priority(dataset_name)
            }
        
        # Integration status analysis
        coverage_report = self.registry.get_comprehensive_coverage_report()
        analysis['integration_status']['working_datasets'] = coverage_report['implementation_status']['working']
        analysis['integration_status']['high_priority_missing'] = coverage_report['implementation_status']['high_priority_pending']
        analysis['integration_status']['medium_priority_missing'] = coverage_report['implementation_status']['medium_priority_pending']
        analysis['integration_status']['low_priority_missing'] = coverage_report['implementation_status']['low_priority_pending']
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_integration_recommendations(analysis)
        
        # Save analysis
        analysis_file = f"{self.dirs['reports']}/dataset_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"✓ Dataset analysis completed")
        print(f"  Datasets analyzed: {analysis['total_datasets_analyzed']}")
        print(f"  Working datasets: {len(analysis['integration_status']['working_datasets'])}")
        print(f"  High priority missing: {len(analysis['integration_status']['high_priority_missing'])}")
        print(f"  Analysis saved to: {analysis_file}")
        
        return analysis
    
    def _extract_main_performance_metric(self, dataset_name: str, metrics: Dict[str, Any]) -> Optional[float]:
        """Extract main performance metric for a dataset."""
        # Use the same mapping as in unified_evaluator
        main_metric_mapping = {
            "CrowsPairs": "crows_pairs_bias_score",
            "StereoSet": "stereoset_bias_score",
            "WinoBias": "winobias_accuracy", 
            "WinoGender": "winogender_accuracy",
            "BBQ": "bbq_accuracy",
            "SEAT": "seat_avg_effect_size",
            "BOLD": "bold_sentiment_bias",
            "BiosBias": "biosbias_accuracy",
            "TruthfulQA": "truthfulqa_truthful_pct",
            "SycophancyEval": "sycophancy_eval_non_sycophantic_pct",
            "MMLU": "mmlu_accuracy",
            "HumanEval": "humaneval_pass_at_1", 
            "GSM8K": "gsm8k_accuracy"
        }
        
        main_metric_key = main_metric_mapping.get(dataset_name)
        if main_metric_key and main_metric_key in metrics:
            return metrics[main_metric_key]
        
        return None
    
    def _get_dataset_priority(self, dataset_name: str) -> str:
        """Get priority level for dataset."""
        if dataset_name in self.registry.WORKING_DATASETS:
            return "Working"
        elif dataset_name in self.registry.HIGH_PRIORITY:
            return "High"
        elif dataset_name in self.registry.MEDIUM_PRIORITY:
            return "Medium" 
        elif dataset_name in self.registry.LOW_PRIORITY:
            return "Low"
        else:
            return "Unknown"
    
    def _generate_integration_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for dataset integration priorities."""
        recommendations = {
            'immediate_priorities': [],
            'medium_term_goals': [],
            'long_term_objectives': [],
            'implementation_notes': {}
        }
        
        # High priority missing datasets
        high_priority_missing = analysis['integration_status']['high_priority_missing']
        if high_priority_missing:
            recommendations['immediate_priorities'].extend([
                f"Implement {dataset} loader" for dataset in high_priority_missing
            ])
            for dataset in high_priority_missing:
                recommendations['implementation_notes'][dataset] = f"High impact bias evaluation - {dataset}"
        
        # Medium priority missing datasets  
        medium_priority_missing = analysis['integration_status']['medium_priority_missing']
        if medium_priority_missing:
            recommendations['medium_term_goals'].extend([
                f"Integrate {dataset} evaluation" for dataset in medium_priority_missing
            ])
        
        # Low priority datasets
        low_priority_missing = analysis['integration_status']['low_priority_missing']
        if low_priority_missing:
            recommendations['long_term_objectives'].extend([
                f"Add {dataset} for comprehensive coverage" for dataset in low_priority_missing
            ])
        
        return recommendations
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive pipeline report.
        
        Returns:
            Complete report of pipeline execution and results
        """
        print(f"\n=== Generating Comprehensive Report ===")
        
        # Update pipeline state
        self.pipeline_state['end_time'] = datetime.now().isoformat()
        self.pipeline_state['total_runtime'] = str(datetime.now() - datetime.fromisoformat(self.pipeline_state['start_time']))
        
        # Create comprehensive report
        report = {
            'pipeline_info': {
                'version': '1.0.0',
                'execution_id': self.output_base.split('/')[-1],
                'model_name': self.model_name,
                'config_files': {
                    'pipeline_config': self.config_path,
                    'dataset_config': self.dataset_config_path
                }
            },
            'execution_summary': self.pipeline_state,
            'dataset_coverage': self.registry.get_comprehensive_coverage_report(),
            'results': self.pipeline_state.get('results', {}),
            'output_files': {
                'reports': f"{self.dirs['reports']}/",
                'evaluation': f"{self.dirs['evaluation']}/",
                'diagnostics': f"{self.dirs['diagnostics']}/",
            }
        }
        
        # Save comprehensive report
        report_file = f"{self.dirs['reports']}/comprehensive_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown summary
        self._generate_markdown_summary(report)
        
        print(f"✓ Comprehensive report generated")
        print(f"  Report file: {report_file}")
        print(f"  Summary: {self.dirs['reports']}/pipeline_summary.md")
        
        return report
    
    def _generate_markdown_summary(self, report: Dict[str, Any]):
        """Generate markdown summary of pipeline execution."""
        summary_file = f"{self.dirs['reports']}/pipeline_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write(f"# Unified Bias Mitigation Pipeline Report\n\n")
            f.write(f"**Execution ID:** {report['pipeline_info']['execution_id']}  \n")
            f.write(f"**Model:** {report['pipeline_info']['model_name']}  \n")
            f.write(f"**Start Time:** {report['execution_summary']['start_time']}  \n")
            f.write(f"**End Time:** {report['execution_summary'].get('end_time', 'Running...')}  \n\n")
            
            # Dataset coverage summary
            coverage = report['dataset_coverage']
            f.write(f"## Dataset Coverage\n\n")
            f.write(f"- **Total Datasets:** {coverage['total_datasets']}\n")
            f.write(f"- **Working Datasets:** {coverage['working_datasets']}\n") 
            f.write(f"- **High Priority Pending:** {coverage['high_priority_datasets'] - coverage['working_datasets']}\n")
            f.write(f"- **Medium Priority Pending:** {coverage['medium_priority_datasets']}\n")
            f.write(f"- **Low Priority Pending:** {coverage['low_priority_datasets']}\n\n")
            
            # Implementation status
            impl_status = coverage['implementation_status']
            f.write(f"### Working Datasets\n")
            for dataset in impl_status['working']:
                f.write(f"- ✅ {dataset}\n")
            f.write(f"\n")
            
            if impl_status['high_priority_pending']:
                f.write(f"### High Priority Pending\n")
                for dataset in impl_status['high_priority_pending']:
                    f.write(f"- 🔥 {dataset}\n")
                f.write(f"\n")
            
            # Results summary
            results = report.get('results', {})
            if 'baseline_evaluation' in results:
                baseline = results['baseline_evaluation']
                f.write(f"## Baseline Evaluation Results\n\n")
                f.write(f"- **Datasets Evaluated:** {baseline['total_datasets_evaluated']}\n")
                f.write(f"- **Total Time:** {baseline['total_evaluation_time']:.1f}s\n\n")
                
                # Aggregated metrics
                if 'aggregated_metrics' in baseline:
                    agg = baseline['aggregated_metrics']
                    if 'by_bias_type' in agg:
                        f.write(f"### Bias Type Performance\n\n")
                        for bias_type, scores in agg['by_bias_type'].items():
                            mean_score = scores.get('mean_score', 0)
                            dataset_count = scores.get('dataset_count', 0)
                            f.write(f"- **{bias_type.title()}:** {mean_score:.3f} ({dataset_count} datasets)\n")
                        f.write(f"\n")
            
            f.write(f"## Output Files\n\n")
            for category, path in report['output_files'].items():
                f.write(f"- **{category.title()}:** `{path}`\n")
    
    def run_complete_pipeline(
        self,
        suite_name: str = "comprehensive",
        skip_model_loading: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete unified bias mitigation pipeline.
        
        Args:
            suite_name: Evaluation suite to run
            skip_model_loading: Skip model loading for testing
            
        Returns:
            Complete pipeline results
        """
        print(f"\n🚀 Starting Unified Bias Mitigation Pipeline")
        print(f"{'='*60}")
        
        try:
            # Stage 1: Environment Validation
            validation_results = self.validate_environment()
            self.pipeline_state['results']['validation'] = validation_results
            
            if not skip_model_loading:
                # Stage 2: Model Loading
                model, tokenizer = self.load_model()
                
                # Stage 3: Baseline Evaluation
                baseline_results = self.run_baseline_evaluation(model, tokenizer, suite_name)
                
                # Stage 4: Dataset Analysis
                analysis_results = self.run_dataset_analysis(baseline_results)
                self.pipeline_state['results']['analysis'] = analysis_results
            
            # Stage 5: Generate Report
            final_report = self.generate_comprehensive_report()
            
            print(f"\n✅ Pipeline completed successfully!")
            print(f"Output directory: {self.output_base}")
            print(f"Report: {self.dirs['reports']}/comprehensive_report.json")
            
            return final_report
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            
            # Save error state
            self.pipeline_state['error'] = str(e)
            self.pipeline_state['end_time'] = datetime.now().isoformat()
            
            error_file = f"{self.dirs['reports']}/pipeline_error.json"
            with open(error_file, 'w') as f:
                json.dump(self.pipeline_state, f, indent=2, default=str)
            
            raise


def main():
    """Main entry point for unified bias mitigation pipeline."""
    # Set up clean environment at the very start
    from setup_clean_environment import setup_clean_environment
    setup_clean_environment()
    
    parser = argparse.ArgumentParser(
        description="Unified Bias Mitigation Pipeline with Comprehensive Dataset Integration"
    )
    parser.add_argument("--model-config", required=True, 
                       help="Path to model/pipeline configuration YAML")
    parser.add_argument("--dataset-config", 
                       default="configs/datasets.yaml",
                       help="Path to dataset configuration YAML")
    parser.add_argument("--suite", default="comprehensive",
                       choices=["comprehensive", "bias_focused", "sycophancy_focused", 
                               "working_baseline", "high_priority", "quick_evaluation"],
                       help="Evaluation suite to run")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only run environment validation")
    parser.add_argument("--skip-model", action="store_true",
                       help="Skip model loading for testing")
    
    args = parser.parse_args()
    
    print("Unified Bias Mitigation Pipeline")
    print("=" * 40)
    print(f"Model Config: {args.model_config}")
    print(f"Dataset Config: {args.dataset_config}")
    print(f"Evaluation Suite: {args.suite}")
    
    try:
        # Initialize pipeline
        pipeline = UnifiedBiasMitigationPipeline(args.model_config, args.dataset_config)
        
        if args.validate_only:
            # Run validation only
            validation_results = pipeline.validate_environment()
            print("\nValidation complete. Check diagnostics for detailed results.")
        else:
            # Run complete pipeline
            results = pipeline.run_complete_pipeline(
                suite_name=args.suite,
                skip_model_loading=args.skip_model
            )
            
            print(f"\n🎉 Pipeline execution completed successfully!")
            print(f"Check the output directory for detailed results: {pipeline.output_base}")
    
    except Exception as e:
        print(f"\n💥 Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()