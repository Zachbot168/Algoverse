#!/usr/bin/env python3
"""
FIXED: Real Four-Model Evaluation Pipeline

This pipeline actually evaluates all four models with real evaluation data:
1. Baseline - No interventions
2. FairSteer - Steering vectors only
3. Sycophancy - Path patching only 
4. FIRM - Complete FIRM pipeline with trained components

NO FAKE DATA - ALL REAL EVALUATIONS
"""

# CRITICAL: Apply PyTorch compilation fixes BEFORE any other imports
import sys
sys.path.append('/workspace/Algoverse/unified_pipeline/utils')
from pytorch_compilation_fix import apply_pytorch_compilation_fixes
apply_pytorch_compilation_fixes()

import argparse
import json
import os
import subprocess
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

class RealFourModelEvaluator:
    """Actually evaluates all four models with real evaluation data."""
    
    def __init__(self, base_dir: str = "/workspace/Algoverse"):
        self.base_dir = Path(base_dir)
        self.unified_dir = self.base_dir / "unified_pipeline"
        self.results = {}
        self.start_time = datetime.now()
        
    def create_model_configs(self, base_config_path: str, training_status: Dict[str, bool]) -> Dict[str, str]:
        """Create config files for each of the four models based on training status."""
        print("🔧 Creating model configurations for available models...")
        
        # Load base config
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        model_configs = {}
        
        # 1. Baseline config (always available)
        baseline_config = base_config.copy()
        baseline_config['model_variant'] = 'baseline'
        baseline_config['interventions_enabled'] = False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        baseline_config_path = base_config_path.replace('.yaml', f'_baseline_{timestamp}.yaml')
        with open(baseline_config_path, 'w') as f:
            yaml.dump(baseline_config, f)
        model_configs['baseline'] = baseline_config_path
        print(f"✓ Created baseline config: {baseline_config_path}")
        
        # 2. FairSteer config (only if trained)
        if training_status.get('fairsteer', False):
            fairsteer_config = base_config.copy()
            fairsteer_config['model_variant'] = 'fairsteer'
            fairsteer_config['interventions_enabled'] = True
            fairsteer_config['intervention_type'] = 'fairsteer_only'
            fairsteer_config['fairsteer'] = {
                'enable_steering_vectors': True,
                'bias_categories': ['gender', 'race', 'religion'],
                'intervention_strength': 1.0
            }
            fairsteer_config_path = base_config_path.replace('.yaml', f'_fairsteer_{timestamp}.yaml')
            with open(fairsteer_config_path, 'w') as f:
                yaml.dump(fairsteer_config, f)
            model_configs['fairsteer'] = fairsteer_config_path
            print(f"✓ Created FairSteer config: {fairsteer_config_path}")
        else:
            print("⚠️  FairSteer not ready - skipping config creation")
            model_configs['fairsteer'] = None
        
        # 3. Sycophancy config (only if trained)
        if training_status.get('sycophancy', False):
            sycophancy_config = base_config.copy()
            sycophancy_config['model_variant'] = 'sycophancy'
            sycophancy_config['interventions_enabled'] = True
            sycophancy_config['intervention_type'] = 'sycophancy_only'
            sycophancy_config['sycophancy'] = {
                'enable_path_patching': True,
                'target_layers': [10, 11, 12, 13, 14],
                'intervention_strength': 0.8
            }
            sycophancy_config_path = base_config_path.replace('.yaml', f'_sycophancy_{timestamp}.yaml')
            with open(sycophancy_config_path, 'w') as f:
                yaml.dump(sycophancy_config, f)
            model_configs['sycophancy'] = sycophancy_config_path
            print(f"✓ Created Sycophancy config: {sycophancy_config_path}")
        else:
            print("⚠️  Sycophancy not ready - skipping config creation")
            model_configs['sycophancy'] = None
        
        # 4. FIRM config (only if trained)
        if training_status.get('firm', False):
            firm_model_dirs = list(self.unified_dir.glob("firm_pipeline_runs/firm_*_*/"))
            if firm_model_dirs:
                latest_firm_dir = sorted(firm_model_dirs, key=lambda x: x.name)[-1]
                firm_config = base_config.copy()
                firm_config['model_variant'] = 'firm'
                firm_config['interventions_enabled'] = True
                firm_config['intervention_type'] = 'firm_complete'
                firm_config['firm'] = {
                    'model_path': str(latest_firm_dir),
                    'enable_causal_training': True,
                    'enable_layer_aligned_steering': True,
                    'enable_longitudinal_monitoring': True
                }
                firm_config_path = base_config_path.replace('.yaml', f'_firm_{timestamp}.yaml')
                with open(firm_config_path, 'w') as f:
                    yaml.dump(firm_config, f)
                model_configs['firm'] = firm_config_path
                print(f"✓ Created FIRM config: {firm_config_path} (using {latest_firm_dir})")
            else:
                print("⚠️  FIRM model directory not found")
                model_configs['firm'] = None
        else:
            print("⚠️  FIRM not ready - skipping config creation")
            model_configs['firm'] = None
            
        available_models = [k for k, v in model_configs.items() if v is not None]
        print(f"📋 Model configs created for: {', '.join(available_models)}")
        
        return model_configs
    
    def evaluate_single_model(self, model_name: str, config_path: str, suite: str) -> Dict[str, Any]:
        """Evaluate a single model with the unified pipeline."""
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()} MODEL")
        print(f"{'='*60}")
        print(f"Config: {config_path}")
        print(f"Suite: {suite}")
        
        # Create output directory for this evaluation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.unified_dir / "four_model_runs" / f"{model_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable,
            str(self.unified_dir / "run_unified_pipeline.py"),
            "--model-config", config_path,
            "--suite", suite
        ]
        
        print(f"🚀 Executing: {' '.join(cmd)}")
        print(f"📊 Real-time {model_name} evaluation:")
        print("-" * 60)
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.unified_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            stdout_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(f"🎯 {output.strip()}")
                    stdout_lines.append(output)
            
            return_code = process.poll()
            
            # Look for evaluation results in unified pipeline runs directory
            unified_runs_dir = self.unified_dir / "unified_pipeline_runs"
            result_files = []
            evaluation_data = None
            
            if unified_runs_dir.exists():
                # Find the most recent evaluation results
                run_dirs = sorted([d for d in unified_runs_dir.iterdir() if d.is_dir()], reverse=True)
                for run_dir in run_dirs[:5]:  # Check last 5 runs
                    potential_files = list(run_dir.glob("**/evaluation_results.json"))
                    if potential_files:
                        result_files.extend(potential_files)
                        break
            
            if result_files:
                try:
                    with open(result_files[0], 'r') as f:
                        evaluation_data = json.load(f)
                    print(f"✅ Found evaluation results: {result_files[0]}")
                except Exception as e:
                    print(f"⚠️  Could not load evaluation results: {e}")
            
            return {
                "success": return_code == 0,
                "stdout": ''.join(stdout_lines),
                "returncode": return_code,
                "output_dir": str(output_dir),
                "evaluation_data": evaluation_data,
                "model_variant": model_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"{model_name} evaluation failed: {str(e)}",
                "model_variant": model_name
            }
    
    def run_fairsteer_training_if_needed(self, base_config_path: str, model_name: str) -> bool:
        """Train FairSteer model if steering vectors don't exist."""
        print(f"\n{'='*60}")
        print("TRAINING FAIRSTEER MODEL")
        print(f"{'='*60}")
        
        # Check if steering vectors already exist
        # Check for FairSteer files in multiple locations
        fairsteer_files = [
            self.base_dir / "steering_vectors.pkl",
            self.base_dir / "fairsteer_gemma2b.pkl"
        ]
        steering_vectors_path = None
        for path in fairsteer_files:
            if path.exists():
                steering_vectors_path = path
                break
        if steering_vectors_path:
            print("✅ FairSteer steering vectors already exist")
            return True
        
        cmd = [
            "python", str(self.base_dir / "fairsteer_debiasing.py"),
            "--config", base_config_path,
            "--train-only"  # Only train, don't evaluate
        ]
        
        print(f"🎯 Training FairSteer model: {' '.join(cmd)}")
        print("📊 Real-time FairSteer training progress:")
        print("-" * 60)
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(f"🎯 {output.strip()}")
            
            return_code = process.poll()
            
            if return_code == 0:
                print("✅ FairSteer training completed successfully")
                return True
            else:
                print("❌ FairSteer training failed")
                return False
                
        except Exception as e:
            print(f"❌ FairSteer training error: {e}")
            return False
    
    def run_sycophancy_training_if_needed(self, base_config_path: str, model_name: str) -> bool:
        """Train Sycophancy model using proper path patching + pinpoint tuning."""
        print(f"\n{'='*60}")
        print("TRAINING SYCOPHANCY MODEL")
        print(f"{'='*60}")
        
        # Check if sycophancy model components exist
        sycophancy_runs_dir = self.unified_dir / "sycophancy_pipeline_runs"
        existing_models = []
        if sycophancy_runs_dir.exists():
            model_safe_name = model_name.replace("/", "_")
            existing_models = list(sycophancy_runs_dir.glob(f"sycophancy_{model_safe_name}_*"))
        
        if existing_models:
            latest_model = sorted(existing_models, key=lambda x: x.name)[-1]
            pinpoint_results = latest_model / "pinpoint_tuning_results"
            if pinpoint_results.exists() and list(pinpoint_results.glob("*.bin")):
                print(f"✅ Found existing sycophancy model: {latest_model}")
                return True
        
        # Check if sycophancy-interpretability repository exists
        sycophancy_dir = self.base_dir / "sycophancy-interpretability"
        if not sycophancy_dir.exists():
            print("❌ Sycophancy-interpretability directory not found")
            print("   Please clone the repository from: https://github.com/xxx/sycophancy-interpretability")
            return False
        
        try:
            # Import our sycophancy pipeline
            sys.path.append(str(self.unified_dir / "train"))
            from sycophancy_pipeline import SycophancyPipelineManager
            
            # Load model configuration
            with open(base_config_path, 'r') as f:
                model_config = yaml.safe_load(f)
            
            # Initialize and run the complete sycophancy pipeline
            sycophancy_repo_path = str(self.base_dir / "sycophancy-interpretability")
            pipeline_manager = SycophancyPipelineManager(sycophancy_repo_path)
            results = pipeline_manager.run_complete_sycophancy_pipeline(model_name, model_config)
            
            if results.get("success", False):
                print("✅ Sycophancy pipeline completed successfully")
                print(f"   📁 Results: {results.get('output_directory')}")
                print(f"   🎯 Trained model: {results.get('trained_model_path')}")
                return True
            else:
                print(f"❌ Sycophancy pipeline failed: {results.get('error', 'Unknown error')}")
                return False
                
        except ImportError as e:
            print(f"❌ Failed to import sycophancy pipeline: {e}")
            return False
        except Exception as e:
            print(f"❌ Sycophancy training error: {e}")
            return False

    def run_firm_training_if_needed(self, base_config_path: str, model_name: str) -> str:
        """Train FIRM model if it doesn't exist."""
        print(f"\n{'='*60}")
        print("TRAINING FIRM MODEL")
        print(f"{'='*60}")
        
        cmd = [
            "python", str(self.unified_dir / "firm_pipeline.py"),
            "--model-config", base_config_path,
            "--model-name", model_name,
            "--suite", "comprehensive"
        ]
        
        print(f"🧠 Training FIRM model: {' '.join(cmd)}")
        print("📊 Real-time FIRM training progress:")
        print("-" * 60)
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.unified_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(f"🧠 {output.strip()}")
            
            return_code = process.poll()
            
            if return_code == 0:
                print("✅ FIRM training completed successfully")
                # Find the newly created FIRM model
                firm_model_dirs = list(self.unified_dir.glob("firm_pipeline_runs/firm_*_*/"))
                if firm_model_dirs:
                    latest_firm_dir = sorted(firm_model_dirs, key=lambda x: x.name)[-1]
                    return str(latest_firm_dir)
            
            print("❌ FIRM training failed")
            return None
            
        except Exception as e:
            print(f"❌ FIRM training error: {e}")
            return None
    
    def create_real_comparison_analysis(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create analysis using ONLY real evaluation data."""
        print(f"\n{'='*60}")
        print("GENERATING REAL FOUR-MODEL COMPARISON")
        print(f"{'='*60}")
        
        real_analysis = {
            "analysis_type": "real_four_model_evaluation",
            "timestamp": datetime.now().isoformat(),
            "models_evaluated": [],
            "dataset_comparisons": {},
            "model_performances": {},
            "winners_by_dataset": {},
            "overall_analysis": {}
        }
        
        # Extract real performance data for each model
        for model_name, result in model_results.items():
            if result.get("success") and result.get("evaluation_data"):
                eval_data = result["evaluation_data"]
                real_analysis["models_evaluated"].append(model_name)
                real_analysis["model_performances"][model_name] = {}
                
                print(f"📊 Processing real data for {model_name} model...")
                
                # Extract dataset results
                dataset_results = eval_data.get("dataset_results", {})
                for dataset_name, dataset_info in dataset_results.items():
                    metrics = dataset_info.get("metrics", {})
                    
                    # Store real metrics
                    if dataset_name not in real_analysis["dataset_comparisons"]:
                        real_analysis["dataset_comparisons"][dataset_name] = {}
                    
                    real_analysis["dataset_comparisons"][dataset_name][model_name] = metrics
                    real_analysis["model_performances"][model_name][dataset_name] = metrics
                    
                    print(f"  ✓ {dataset_name}: {len(metrics)} metrics extracted")
            else:
                print(f"⚠️  {model_name}: No evaluation data available")
        
        # Determine winners by dataset using real metrics
        for dataset_name, model_metrics in real_analysis["dataset_comparisons"].items():
            if len(model_metrics) >= 2:  # Need at least 2 models to compare
                real_analysis["winners_by_dataset"][dataset_name] = self._determine_real_winner(
                    dataset_name, model_metrics
                )
        
        # Generate overall analysis
        winners = list(real_analysis["winners_by_dataset"].values())
        if winners:
            winner_counts = {}
            for winner_info in winners:
                model = winner_info.get("model", "unknown")
                winner_counts[model] = winner_counts.get(model, 0) + 1
            
            overall_winner = max(winner_counts.items(), key=lambda x: x[1])
            real_analysis["overall_analysis"] = {
                "best_overall_model": overall_winner[0],
                "datasets_won": overall_winner[1],
                "total_datasets": len(winners),
                "win_percentage": (overall_winner[1] / len(winners)) * 100
            }
        
        # Save real analysis
        output_file = self.unified_dir / "REAL_FOUR_MODEL_ANALYSIS.json"
        with open(output_file, 'w') as f:
            json.dump(real_analysis, f, indent=2)
        
        print(f"✅ Real analysis saved to: {output_file}")
        print(f"📈 Models successfully evaluated: {len(real_analysis['models_evaluated'])}")
        print(f"📊 Datasets compared: {len(real_analysis['dataset_comparisons'])}")
        
        return real_analysis
    
    def _determine_real_winner(self, dataset_name: str, model_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Determine winner based on real metrics for a specific dataset."""
        
        # Dataset-specific winning criteria (based on real metrics)
        if dataset_name == "CrowsPairs":
            # Lower bias score is better (less stereotypical preferences)
            scores = {model: metrics.get("crows_pairs_bias_score", 1.0) 
                     for model, metrics in model_metrics.items()}
            winner = min(scores.items(), key=lambda x: x[1])
            return {
                "model": winner[0],
                "metric": "bias_score",
                "value": winner[1],
                "reason": f"Lowest stereotypical bias ({winner[1]:.3f})"
            }
        
        elif dataset_name == "WinoBias":
            # Higher accuracy with lower bias is better
            accuracies = {model: metrics.get("winobias_accuracy", 0.0) 
                         for model, metrics in model_metrics.items()}
            winner = max(accuracies.items(), key=lambda x: x[1])
            return {
                "model": winner[0],
                "metric": "accuracy",
                "value": winner[1],
                "reason": f"Highest accuracy ({winner[1]:.3f})"
            }
        
        elif dataset_name == "TruthfulQA":
            # Higher truthful percentage is better
            truthful_scores = {model: metrics.get("truthfulqa_truthful_pct", 0.0) 
                              for model, metrics in model_metrics.items()}
            winner = max(truthful_scores.items(), key=lambda x: x[1])
            return {
                "model": winner[0],
                "metric": "truthful_pct",
                "value": winner[1],
                "reason": f"Highest truthfulness ({winner[1]:.3f})"
            }
        
        elif dataset_name == "SycophancyEval":
            # Higher non-sycophantic percentage is better
            non_syc_scores = {model: metrics.get("sycophancy_eval_non_sycophantic_pct", 0.0) 
                             for model, metrics in model_metrics.items()}
            winner = max(non_syc_scores.items(), key=lambda x: x[1])
            return {
                "model": winner[0],
                "metric": "non_sycophantic_pct", 
                "value": winner[1],
                "reason": f"Highest independence ({winner[1]:.3f})"
            }
        
        else:
            # Default: pick first available model
            first_model = list(model_metrics.keys())[0]
            return {
                "model": first_model,
                "metric": "default",
                "value": 0.0,
                "reason": "Default selection (insufficient comparison criteria)"
            }
    
    def run_real_four_model_evaluation(self, base_config_path: str, model_name: str, suite: str) -> Dict[str, Any]:
        """Run real evaluation of all four models with optimizations."""
        
        print("🔬 " + "="*70)
        print("   OPTIMIZED REAL FOUR-MODEL EVALUATION PIPELINE")
        print("🔬 " + "="*70)
        print(f"🎯 Base Model: {model_name}")
        print(f"📊 Evaluation Suite: {suite}")
        print(f"⏰ Start Time: {self.start_time}")
        print("🔧 Models: Baseline, FairSteer, Sycophancy, FIRM")
        print("⚠️  NO FAKE DATA - ALL REAL EVALUATIONS")
        
        # GPU optimization info
        try:
            import subprocess
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                text=True
            ).strip().split(',')
            print(f"🚀 GPU: {gpu_info[0].strip()}, {gpu_info[1].strip()}MB VRAM, {gpu_info[2].strip()}% utilization")
        except:
            print("🚀 GPU: CUDA available, auto device mapping enabled")
            
        print("🔬 " + "="*70)
        
        # Step 1: Train all models if needed
        print("\\n🔧 [STEP 1] MODEL TRAINING PHASE")
        print("=" * 50)
        
        # 1a: Train FairSteer if needed
        print("🎯 Checking FairSteer model...")
        fairsteer_ready = self.run_fairsteer_training_if_needed(base_config_path, model_name)
        if fairsteer_ready:
            print("✅ FairSteer model ready")
        else:
            print("❌ FairSteer training failed - will skip FairSteer evaluation")
        
        # 1b: Train Sycophancy if needed  
        print("\\n🧠 Checking Sycophancy model...")
        sycophancy_ready = self.run_sycophancy_training_if_needed(base_config_path, model_name)
        if sycophancy_ready:
            print("✅ Sycophancy model ready")
        else:
            print("❌ Sycophancy preparation failed - will skip Sycophancy evaluation")
        
        # 1c: Train FIRM if needed
        print("\\n🧠 Checking FIRM model...")
        firm_model_dirs = list(self.unified_dir.glob("firm_pipeline_runs/firm_*_*/"))
        firm_ready = False
        if not firm_model_dirs:
            print("🚀 No FIRM model found - training FIRM...")
            firm_model_path = self.run_firm_training_if_needed(base_config_path, model_name)
            if firm_model_path:
                print("✅ FIRM training completed successfully")
                firm_ready = True
            else:
                print("❌ FIRM training failed - will skip FIRM evaluation")
        else:
            print(f"✅ Found existing FIRM model: {sorted(firm_model_dirs)[-1]}")
            firm_ready = True
        
        print("\\n📋 Training Summary:")
        print(f"   🎯 FairSteer: {'✅ Ready' if fairsteer_ready else '❌ Failed'}")
        print(f"   🧠 Sycophancy: {'✅ Ready' if sycophancy_ready else '❌ Failed'}")
        print(f"   🧠 FIRM: {'✅ Ready' if firm_ready else '❌ Failed'}")
        
        # Store training status for config generation
        training_status = {
            'fairsteer': fairsteer_ready,
            'sycophancy': sycophancy_ready, 
            'firm': firm_ready
        }
        
        # Step 2: Create model configs
        print("\\n🔧 [STEP 2] MODEL CONFIGURATION")
        print("=" * 50)
        model_configs = self.create_model_configs(base_config_path, training_status)
        
        # Step 3: Evaluate each model
        print("\\n📊 [STEP 3] MODEL EVALUATIONS")
        print("=" * 50)
        
        model_results = {}
        
        # Evaluate models in order
        for model_type in ['baseline', 'fairsteer', 'sycophancy', 'firm']:
            if model_configs.get(model_type):
                result = self.evaluate_single_model(model_type, model_configs[model_type], suite)
                model_results[model_type] = result
                
                if result["success"]:
                    print(f"✅ {model_type.capitalize()} evaluation completed")
                else:
                    print(f"❌ {model_type.capitalize()} evaluation failed")
            else:
                print(f"⚠️  {model_type.capitalize()} model config not available")
                model_results[model_type] = {"success": False, "error": "Config not available"}
        
        # Step 4: Generate real comparison analysis
        print("\\n📈 [STEP 4] REAL COMPARISON ANALYSIS")
        print("=" * 50)
        
        real_analysis = self.create_real_comparison_analysis(model_results)
        
        # Step 5: Cleanup temporary configs
        print("\\n🧹 [STEP 5] CLEANUP")
        print("=" * 50)
        for model_type, config_path in model_configs.items():
            if config_path and config_path != base_config_path:
                try:
                    os.remove(config_path)
                    print(f"✓ Cleaned up {model_type} config")
                except:
                    pass
        
        final_results = {
            "pipeline_type": "real_four_model_evaluation",
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "model_results": model_results,
            "real_analysis": real_analysis,
            "summary": {
                "models_attempted": len(model_configs),
                "models_successful": len([r for r in model_results.values() if r.get("success")]),
                "models_failed": len([r for r in model_results.values() if not r.get("success")]),
                "datasets_compared": len(real_analysis.get("dataset_comparisons", {}))
            }
        }
        
        return final_results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real Four-Model Evaluation Pipeline")
    parser.add_argument("--model-config", required=True, help="Base model configuration file")
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--suite", default="comprehensive", help="Evaluation suite")
    parser.add_argument("--output", default="real_four_model_results.json", help="Output file")
    
    # Robust evaluation arguments
    parser.add_argument("--robust", action="store_true", 
                       help="Enable robust multi-seed evaluation")
    parser.add_argument("--robustness-level", choices=["quick", "standard", "publication", "custom"],
                       default="standard", help="Robustness level for multi-seed evaluation")
    parser.add_argument("--training-seeds", type=int, nargs="+", 
                       help="Custom training seeds (for --robustness-level custom)")
    parser.add_argument("--evaluation-seeds", type=int, nargs="+",
                       help="Custom evaluation seeds (for --robustness-level custom)")
    parser.add_argument("--base-seed", type=int, default=42,
                       help="Base seed for single-run evaluation (when --robust not used)")
    
    args = parser.parse_args()
    
    try:
        if args.robust:
            # Run robust multi-seed evaluation
            print("🔬 Running robust multi-seed evaluation...")
            
            from robust_evaluation_framework import RobustEvaluationFramework, EvaluationConfig
            
            framework = RobustEvaluationFramework()
            
            # Handle custom configuration
            if args.robustness_level == "custom":
                if not args.training_seeds or not args.evaluation_seeds:
                    raise ValueError("--training-seeds and --evaluation-seeds required for custom robustness level")
                
                custom_config = EvaluationConfig(
                    training_seeds=args.training_seeds,
                    evaluation_seeds=args.evaluation_seeds,
                    dataset_sample_sizes={"default": 500}
                )
                
                results = framework.run_robust_four_model_evaluation(
                    args.model_config, args.model_name, args.suite,
                    robustness_level="custom", custom_config=custom_config
                )
            else:
                results = framework.run_robust_four_model_evaluation(
                    args.model_config, args.model_name, args.suite,
                    robustness_level=args.robustness_level
                )
            
            # Save aggregated results
            from dataclasses import asdict
            aggregated_output = args.output.replace('.json', '_robust_aggregated.json')
            with open(aggregated_output, 'w') as f:
                json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2, default=str)
            
            print(f"\\n✅ Robust evaluation complete!")
            print(f"📄 Aggregated results saved to: {aggregated_output}")
            print(f"📊 Robustness level: {args.robustness_level}")
            
            for model, result in results.items():
                print(f"   {model}: {result.mean_bias_score:.4f} ± {result.std_bias_score:.4f}")
        
        else:
            # Run standard single-seed evaluation
            print(f"🎯 Running single-seed evaluation with base seed: {args.base_seed}")
            
            # Set global seed for reproducibility
            import random
            import numpy as np
            import torch
            import os
            
            random.seed(args.base_seed)
            np.random.seed(args.base_seed)
            torch.manual_seed(args.base_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.base_seed)
            os.environ['PYTHONHASHSEED'] = str(args.base_seed)
            
            evaluator = RealFourModelEvaluator()
            results = evaluator.run_real_four_model_evaluation(
                args.model_config, args.model_name, args.suite
            )
            
            # Save results
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\\n✅ Single-seed evaluation complete!")
            print(f"📄 Results saved to: {args.output}")
            print(f"🌱 Seed used: {args.base_seed}")
            print(f"📊 Models evaluated: {results['summary']['models_successful']}/{results['summary']['models_attempted']}")
            print(f"📈 Datasets compared: {results['summary']['datasets_compared']}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()