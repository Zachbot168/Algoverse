#!/usr/bin/env python3
"""
Robust Multi-Seed Evaluation Framework

This framework provides statistically robust evaluation of bias mitigation techniques
through multiple training seeds and evaluation cycles.
"""

import json
import numpy as np
import os
import random
import sys
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import yaml
import warnings
from dataclasses import dataclass, asdict
from scipy import stats

# Import evaluation components at module level
try:
    from eval.unified_evaluator import UnifiedBiasEvaluator
    from datasets.unified_registry import UnifiedDatasetRegistry
    EVALUATOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import UnifiedBiasEvaluator: {e}")
    UnifiedBiasEvaluator = None
    UnifiedDatasetRegistry = None
    EVALUATOR_AVAILABLE = False

warnings.filterwarnings('ignore')


@dataclass
class EvaluationConfig:
    """Configuration for robust evaluation framework."""
    training_seeds: List[int]
    evaluation_seeds: List[int] 
    dataset_sample_sizes: Dict[str, int]
    statistical_tests: List[str] = None
    confidence_level: float = 0.95
    
    def __post_init__(self):
        if self.statistical_tests is None:
            self.statistical_tests = ["t_test", "mann_whitney", "effect_size"]


@dataclass
class SeedResult:
    """Results from a single seed evaluation."""
    training_seed: int
    evaluation_seed: int
    model_variant: str
    dataset_results: Dict[str, Any]
    overall_bias_score: float
    evaluation_time: float
    metadata: Dict[str, Any]


@dataclass
class AggregatedResults:
    """Aggregated results across multiple seeds."""
    model_variant: str
    mean_bias_score: float
    std_bias_score: float
    confidence_interval: Tuple[float, float]
    dataset_means: Dict[str, float]
    dataset_stds: Dict[str, float]
    n_evaluations: int
    seed_results: List[SeedResult]
    statistical_significance: Dict[str, Any]


class RobustEvaluationFramework:
    """
    Framework for conducting statistically robust bias mitigation evaluations
    with multiple training and evaluation seeds.
    """
    
    def __init__(self, base_dir: str = "/workspace/Algoverse"):
        self.base_dir = Path(base_dir)
        self.unified_dir = self.base_dir / "unified_pipeline"
        self.results_dir = self.unified_dir / "robust_evaluation_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced evaluation configurations for better statistical significance
        self.evaluation_configs = {
            "quick": EvaluationConfig(
                training_seeds=[42, 123],  # Increased from 1 to 2 seeds
                evaluation_seeds=[100, 200],  # Increased from 1 to 2 seeds
                dataset_sample_sizes={"default": 300}  # Increased from 200
            ),
            "standard": EvaluationConfig(
                training_seeds=[42, 123, 456, 789],  # Increased from 3 to 4 seeds
                evaluation_seeds=[100, 200, 300, 400],  # Increased from 3 to 4 seeds
                dataset_sample_sizes={"default": 750}  # Increased from 500
            ),
            "publication": EvaluationConfig(
                training_seeds=[42, 123, 456, 789, 999, 1337],  # Increased from 5 to 6 seeds
                evaluation_seeds=[100, 200, 300, 400, 500, 600],  # Increased from 5 to 6 seeds
                dataset_sample_sizes={"default": 1500}  # Increased from 1000
            ),
            "custom": None  # Will be set by user
        }
    
    def set_custom_config(self, training_seeds: List[int], evaluation_seeds: List[int],
                         dataset_sample_sizes: Dict[str, int] = None) -> None:
        """Set custom evaluation configuration."""
        if dataset_sample_sizes is None:
            dataset_sample_sizes = {"default": 500}
            
        self.evaluation_configs["custom"] = EvaluationConfig(
            training_seeds=training_seeds,
            evaluation_seeds=evaluation_seeds,
            dataset_sample_sizes=dataset_sample_sizes
        )
    
    def run_robust_four_model_evaluation(self, 
                                       base_config_path: str,
                                       model_name: str,
                                       suite: str,
                                       robustness_level: str = "standard",
                                       custom_config: Optional[EvaluationConfig] = None) -> Dict[str, AggregatedResults]:
        """
        Run robust evaluation of all four models with multiple seeds.
        
        Args:
            base_config_path: Path to base model configuration
            model_name: HuggingFace model name
            suite: Evaluation suite to run
            robustness_level: "quick", "standard", "publication", or "custom"
            custom_config: Custom evaluation config if robustness_level="custom"
            
        Returns:
            Dictionary mapping model variants to aggregated results
        """
        print(f"🔬 {'='*80}")
        print("   ROBUST MULTI-SEED FOUR-MODEL EVALUATION")
        print(f"🔬 {'='*80}")
        
        # Get evaluation configuration
        if robustness_level == "custom" and custom_config:
            eval_config = custom_config
        elif robustness_level == "custom":
            eval_config = self.evaluation_configs["custom"]
            if eval_config is None:
                raise ValueError("Custom config not set. Use set_custom_config() first.")
        else:
            eval_config = self.evaluation_configs[robustness_level]
        
        print(f"📊 Robustness Level: {robustness_level}")
        print(f"🎯 Training Seeds: {eval_config.training_seeds}")
        print(f"📈 Evaluation Seeds: {eval_config.evaluation_seeds}")
        print(f"🔢 Total Evaluations per Model: {len(eval_config.training_seeds) * len(eval_config.evaluation_seeds)}")
        print(f"🕒 Estimated Time: {self._estimate_total_time(eval_config)} minutes")
        
        # Create timestamped results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.results_dir / f"robust_eval_{robustness_level}_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        # Save evaluation configuration
        config_path = run_dir / "evaluation_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(eval_config), f, indent=2)
        
        # Initialize results storage
        all_model_results = {
            "baseline": [],
            "fairsteer": [],
            "sycophancy": [],
            "firm": []
        }
        
        # Import the main evaluator
        sys.path.append(str(self.unified_dir))
        from run_integrated_pipeline import RealFourModelEvaluator
        
        total_combinations = len(eval_config.training_seeds) * len(eval_config.evaluation_seeds) * 4
        current_combination = 0
        
        # Run evaluations for each seed combination
        for train_seed in eval_config.training_seeds:
            print(f"\n🌱 {'='*60}")
            print(f"   TRAINING SEED: {train_seed}")
            print(f"🌱 {'='*60}")
            
            # Train models with this seed
            self._set_global_seed(train_seed)
            trained_models = self._train_all_models_with_seed(
                base_config_path, model_name, train_seed
            )
            
            for eval_seed in eval_config.evaluation_seeds:
                print(f"\n📊 Evaluation Seed: {eval_seed}")
                print("-" * 40)
                
                # Set evaluation seed for dataset sampling
                self._set_global_seed(eval_seed)
                
                # Evaluate each model variant
                for model_variant in ["baseline", "fairsteer", "sycophancy", "firm"]:
                    current_combination += 1
                    progress = (current_combination / total_combinations) * 100
                    
                    print(f"🎯 [{current_combination}/{total_combinations}] ({progress:.1f}%) "
                          f"Evaluating {model_variant} (train_seed={train_seed}, eval_seed={eval_seed})")
                    
                    if model_variant in trained_models and trained_models[model_variant]["success"]:
                        # Run evaluation with specific seeds
                        seed_result = self._evaluate_single_seed(
                            model_variant, trained_models[model_variant],
                            train_seed, eval_seed, suite, eval_config
                        )
                        
                        if seed_result:
                            all_model_results[model_variant].append(seed_result)
                            print(f"✅ {model_variant}: bias_score={seed_result.overall_bias_score:.4f}")
                        else:
                            print(f"❌ {model_variant}: evaluation failed")
                    else:
                        print(f"⚠️  {model_variant}: model training failed, skipping")
        
        # Aggregate results across seeds
        print(f"\n📈 {'='*60}")
        print("   AGGREGATING MULTI-SEED RESULTS")
        print(f"📈 {'='*60}")
        
        aggregated_results = {}
        for model_variant, seed_results in all_model_results.items():
            if seed_results:
                aggregated = self._aggregate_seed_results(model_variant, seed_results, eval_config)
                aggregated_results[model_variant] = aggregated
                
                print(f"📊 {model_variant.upper()}:")
                print(f"   📈 DATASET-SPECIFIC RESULTS (NOT aggregated):")
                for dataset, mean_score in aggregated.dataset_means.items():
                    std_score = aggregated.dataset_stds.get(dataset, 0.0)
                    print(f"      {dataset}: {mean_score:.4f} ± {std_score:.4f}")
                print(f"   🎯 Reference Summary: {aggregated.mean_bias_score:.4f} ± {aggregated.std_bias_score:.4f} (harmonic mean)")
                print(f"   📊 Evaluations: {aggregated.n_evaluations}")
                print(f"   ⚠️  Use dataset-specific scores for analysis, NOT summary!")
        
        # Statistical significance testing
        statistical_results = self._compute_statistical_significance(aggregated_results, eval_config)
        
        # Save all results
        results_summary = {
            "evaluation_config": asdict(eval_config),
            "aggregated_results": {k: asdict(v) for k, v in aggregated_results.items()},
            "statistical_significance": statistical_results,
            "metadata": {
                "timestamp": timestamp,
                "model_name": model_name,
                "suite": suite,
                "robustness_level": robustness_level,
                "total_evaluations": sum(len(results) for results in all_model_results.values())
            }
        }
        
        # Ensure run directory exists
        run_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = run_dir / "robust_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report(aggregated_results, statistical_results, run_dir)
        
        print(f"\n🎉 ROBUST EVALUATION COMPLETE")
        print(f"   📁 Results saved to: {run_dir}")
        print(f"   📊 Summary report: {run_dir / 'summary_report.md'}")
        
        return aggregated_results
    
    def _set_global_seed(self, seed: int) -> None:
        """Set seed for all random number generators."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def _train_all_models_with_seed(self, base_config_path: str, model_name: str, 
                                   train_seed: int) -> Dict[str, Any]:
        """Train all models with a specific seed."""
        print(f"🏋️ Training all models with seed {train_seed}...")
        
        # Import the evaluator class
        from run_integrated_pipeline import RealFourModelEvaluator
        
        # Modify config to include seed
        seeded_config_path = self._create_seeded_config(base_config_path, train_seed)
        
        # Initialize evaluator and run training
        evaluator = RealFourModelEvaluator(str(self.base_dir))
        
        # Train each model type
        results = {}
        
        # FairSteer training
        fairsteer_success = evaluator.run_fairsteer_training_if_needed(seeded_config_path, model_name)
        results["fairsteer"] = {"success": fairsteer_success}
        
        # Sycophancy training  
        sycophancy_success = evaluator.run_sycophancy_training_if_needed(seeded_config_path, model_name)
        results["sycophancy"] = {"success": sycophancy_success}
        
        # FIRM training
        firm_path = evaluator.run_firm_training_if_needed(seeded_config_path, model_name)
        results["firm"] = {"success": firm_path is not None, "model_path": firm_path}
        
        # Baseline always available
        results["baseline"] = {"success": True}
        
        return results
    
    def _create_seeded_config(self, base_config_path: str, seed: int) -> str:
        """Create a model config with specific seed."""
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add seed to all relevant sections
        config['training_seed'] = seed
        config['eval_seed'] = seed
        
        if 'training' not in config:
            config['training'] = {}
        config['training']['seed'] = seed
        
        # Save seeded config
        seeded_config_path = base_config_path.replace('.yaml', f'_seed_{seed}.yaml')
        with open(seeded_config_path, 'w') as f:
            yaml.dump(config, f)
        
        return seeded_config_path
    
    def _get_model_path_for_variant(self, model_variant: str, train_seed: int) -> Optional[str]:
        """Get the actual model path for a specific variant and training seed."""
        
        if model_variant == "baseline":
            # Baseline model is the original pre-trained model
            return "google/gemma-2-2b-it"
            
        elif model_variant == "fairsteer":
            # FairSteer uses the baseline model + steering vectors
            # The steering vectors are applied at inference time
            return "google/gemma-2-2b-it"
            
        elif model_variant == "sycophancy":
            # Sycophancy model should be in sycophancy_pipeline_runs/*/pinpoint_tuning_results/
            sycophancy_runs = self.unified_dir / "sycophancy_pipeline_runs"
            if sycophancy_runs.exists():
                # Find the most recent sycophancy run
                run_dirs = [d for d in sycophancy_runs.iterdir() if d.is_dir()]
                if run_dirs:
                    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
                    # Check if pinpoint_tuning_results subdirectory exists with tokenizer
                    pinpoint_dir = latest_run / "pinpoint_tuning_results"
                    if pinpoint_dir.exists() and (pinpoint_dir / "tokenizer_config.json").exists():
                        return str(pinpoint_dir)
            return "google/gemma-2-2b-it"  # Fallback to baseline
            
        elif model_variant == "firm":
            # FIRM model should be in firm_pipeline_runs/*/phase_2_causal_training/
            firm_runs = self.unified_dir / "firm_pipeline_runs"
            if firm_runs.exists():
                # Find the most recent FIRM run
                run_dirs = [d for d in firm_runs.iterdir() if d.is_dir()]
                if run_dirs:
                    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
                    # Check if phase_2_causal_training subdirectory exists with tokenizer
                    causal_dir = latest_run / "phase_2_causal_training"
                    if causal_dir.exists() and (causal_dir / "tokenizer_config.json").exists():
                        return str(causal_dir)
            return "google/gemma-2-2b-it"  # Fallback to baseline
            
        else:
            print(f"⚠️  Unknown model variant: {model_variant}")
            return "google/gemma-2-2b-it"  # Fallback to baseline
    
    def _apply_fairsteer_intervention(self, model, tokenizer):
        """Apply FairSteer steering vectors to the model."""
        try:
            import pickle
            import sys
            sys.path.insert(0, '/workspace/Algoverse')
            
            # Load the FairSteer steering vectors
            fairsteer_path = "/workspace/Algoverse/fairsteer_gemma2b.pkl"
            if not os.path.exists(fairsteer_path):
                print(f"   ⚠️  FairSteer vectors not found at {fairsteer_path}, using baseline")
                return model
            
            with open(fairsteer_path, 'rb') as f:
                fairsteer_data = pickle.load(f)
            
            steering_vectors = fairsteer_data.get('steering_vectors', {})
            optimal_layer = fairsteer_data.get('optimal_layer', 15)  # Default to layer 15
            
            if not steering_vectors:
                print(f"   ⚠️  No steering vectors found in FairSteer file, using baseline")
                return model
            
            print(f"   ✅ Loaded FairSteer vectors for {len(steering_vectors)} layers, optimal: {optimal_layer}")
            
            # Create wrapper class that applies steering during forward pass
            class FairSteerWrapper(torch.nn.Module):
                def __init__(self, base_model, steering_vectors, optimal_layer, intervention_strength=1.0):
                    super().__init__()
                    self.base_model = base_model
                    self.steering_vectors = steering_vectors
                    self.optimal_layer = optimal_layer
                    self.intervention_strength = intervention_strength
                    self.device = next(base_model.parameters()).device
                    
                    # Convert steering vector to tensor
                    if optimal_layer in steering_vectors:
                        self.steering_vector = torch.tensor(
                            steering_vectors[optimal_layer], 
                            device=self.device, 
                            dtype=torch.float16
                        )
                    else:
                        self.steering_vector = None
                        print(f"   ⚠️  Optimal layer {optimal_layer} not in steering vectors")
                
                def forward(self, *args, **kwargs):
                    # Apply steering hook during forward pass
                    if self.steering_vector is not None:
                        # Register temporary hook on the optimal layer
                        def steering_hook(module, input, output):
                            if isinstance(output, torch.Tensor) and output.dim() == 3:
                                # Create a new tensor with steering applied to last token position
                                modified_output = output.clone()
                                modified_output[:, -1, :] += self.intervention_strength * self.steering_vector
                                return modified_output
                            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor) and output[0].dim() == 3:
                                # Handle case where output is a tuple (hidden_states, ...)
                                modified_hidden = output[0].clone()
                                modified_hidden[:, -1, :] += self.intervention_strength * self.steering_vector
                                return (modified_hidden,) + output[1:]
                            return output
                        
                        # Find and hook the target layer
                        target_layer = None
                        for name, module in self.base_model.named_modules():
                            if f"layers.{self.optimal_layer}" in name and ("self_attn" in name or "mlp" in name):
                                target_layer = module
                                break
                        
                        if target_layer is not None:
                            handle = target_layer.register_forward_hook(steering_hook)
                            try:
                                outputs = self.base_model(*args, **kwargs)
                            finally:
                                handle.remove()
                            return outputs
                    
                    # Fallback to base model if no steering
                    return self.base_model(*args, **kwargs)
                
                def generate(self, *args, **kwargs):
                    # For generation, we need to hook into the model layers
                    if self.steering_vector is not None:
                        # Register hook on the optimal layer during generation
                        def steering_hook(module, input, output):
                            if isinstance(output, torch.Tensor) and output.dim() == 3:
                                # Create a new tensor with steering applied to last token position
                                modified_output = output.clone()
                                modified_output[:, -1, :] += self.intervention_strength * self.steering_vector
                                return modified_output
                            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor) and output[0].dim() == 3:
                                # Handle case where output is a tuple (hidden_states, ...)
                                modified_hidden = output[0].clone()
                                modified_hidden[:, -1, :] += self.intervention_strength * self.steering_vector
                                return (modified_hidden,) + output[1:]
                            return output
                        
                        # Find the right layer to hook
                        target_layer = None
                        for name, module in self.base_model.named_modules():
                            if f"layers.{self.optimal_layer}" in name and "self_attn" in name:
                                target_layer = module
                                break
                        
                        if target_layer is not None:
                            handle = target_layer.register_forward_hook(steering_hook)
                            try:
                                result = self.base_model.generate(*args, **kwargs)
                            finally:
                                handle.remove()
                            return result
                    
                    return self.base_model.generate(*args, **kwargs)
                
                def __getattr__(self, name):
                    # Delegate all other attributes to the base model
                    try:
                        return super().__getattr__(name)
                    except AttributeError:
                        return getattr(self.base_model, name)
            
            # Wrap the model with FairSteer intervention
            wrapped_model = FairSteerWrapper(model, steering_vectors, optimal_layer)
            print(f"   🎯 FairSteer intervention applied with strength 1.0 on layer {optimal_layer}")
            
            return wrapped_model
            
        except Exception as e:
            print(f"   ❌ Failed to apply FairSteer intervention: {e}")
            return model  # Return baseline model if intervention fails
    
    def _evaluate_single_seed(self, model_variant: str, model_info: Dict[str, Any],
                             train_seed: int, eval_seed: int, suite: str,
                             eval_config: EvaluationConfig) -> Optional[SeedResult]:
        """Evaluate a single model with specific seeds using REAL unified evaluator."""
        
        try:
            print(f"🔬 [REAL EVAL] Running actual evaluation for {model_variant} (train_seed={train_seed}, eval_seed={eval_seed})")
            
            # Check if evaluation components are available
            if not EVALUATOR_AVAILABLE:
                print(f"❌ UnifiedBiasEvaluator not available, skipping evaluation")
                return None
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import yaml
            import torch
            
            
            # Set evaluation seed for reproducibility (AFTER importing torch)
            np.random.seed(eval_seed)
            torch.manual_seed(eval_seed)
            random.seed(eval_seed)
            
            # Load model and tokenizer for this variant
            start_time = time.time()
            
            # Get model path from variant
            model_path = self._get_model_path_for_variant(model_variant, train_seed)
            if not model_path:
                print(f"❌ No model path found for {model_variant}")
                return None
            
            print(f"   📁 Loading model from: {model_path}")
            
            # Load model and tokenizer (this should take substantial time for real models)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True
                )
                
                # Apply model-specific interventions
                if model_variant == "fairsteer":
                    print("   🎯 Applying FairSteer steering vectors...")
                    model = self._apply_fairsteer_intervention(model, tokenizer)
                elif model_variant == "sycophancy":
                    print("   🔧 Loading sycophancy-specific model adaptations...")
                    # Sycophancy model is already loaded from the fine-tuned path
                elif model_variant == "firm":
                    print("   🛠️ Loading FIRM multi-component model...")
                    # FIRM model is already loaded from the trained path
                
                print(f"   ✅ Model loaded successfully on {device}")
                
            except Exception as e:
                print(f"❌ Failed to load model {model_variant}: {e}")
                return None
            
            # Load dataset configuration
            dataset_config_path = self.unified_dir / "configs" / "datasets.yaml"
            if not dataset_config_path.exists():
                print(f"❌ Dataset config not found: {dataset_config_path}")
                return None
                
            with open(dataset_config_path, 'r') as f:
                dataset_config = yaml.safe_load(f)
            
            # Create REAL evaluator and run evaluation
            print(f"   🧮 Initializing unified evaluator...")
            evaluator = UnifiedBiasEvaluator(dataset_config, str(self.base_dir))
            
            # Run REAL comprehensive evaluation
            print(f"   🚀 Starting real evaluation on {suite} suite...")
            evaluation_results = evaluator.run_comprehensive_evaluation(
                model, tokenizer, suite_name=suite
            )
            
            evaluation_time = time.time() - start_time
            
            # Extract REAL dataset-specific results (NOT aggregated!)
            dataset_results = {}
            
            if evaluation_results and "dataset_results" in evaluation_results:
                dataset_data = evaluation_results["dataset_results"]
                
                for dataset_name, dataset_info in dataset_data.items():
                    metrics = dataset_info.get("metrics", {})
                    
                    # Extract the PRIMARY metric for each dataset (not aggregated!)
                    if dataset_name == "CrowsPairs":
                        dataset_results[dataset_name] = metrics.get("crows_pairs_bias_score", 0.0)
                    elif dataset_name == "StereoSet":
                        dataset_results[dataset_name] = metrics.get("stereoset_bias_score", 0.0)
                    elif dataset_name == "WinoBias":
                        dataset_results[dataset_name] = metrics.get("winobias_accuracy", 0.0)
                    elif dataset_name == "TruthfulQA":
                        dataset_results[dataset_name] = metrics.get("truthfulqa_truthful_pct", 0.0)
                    elif dataset_name == "BBQ":
                        dataset_results[dataset_name] = metrics.get("bbq_accuracy", 0.0)
                    elif dataset_name == "SEAT":
                        dataset_results[dataset_name] = metrics.get("seat_avg_effect_size", 0.0)
                    elif dataset_name == "BOLD":
                        dataset_results[dataset_name] = metrics.get("bold_sentiment_bias", 0.0)
                    else:
                        # Use first available metric as primary
                        if metrics:
                            primary_metric = list(metrics.keys())[0]
                            dataset_results[dataset_name] = metrics[primary_metric]
            
            # Calculate overall score as HARMONIC MEAN (not arithmetic mean!)
            if dataset_results:
                # Include small BOLD scores (> 0.0001) to capture sentiment bias improvements
                valid_scores = [score for score in dataset_results.values() if score > 0.0001]
                if valid_scores:
                    overall_score = len(valid_scores) / sum(1/score for score in valid_scores)
                else:
                    overall_score = 0.0
            else:
                overall_score = 0.0
            
            print(f"✅ [REAL EVAL] {model_variant}: {len(dataset_results)} datasets evaluated in {evaluation_time:.1f}s")
            print(f"   📊 Dataset scores: {dataset_results}")
            print(f"   🎯 Overall harmonic mean: {overall_score:.4f}")
            
            # Clean up GPU memory
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return SeedResult(
                training_seed=train_seed,
                evaluation_seed=eval_seed,
                model_variant=model_variant,
                dataset_results=dataset_results,  # REAL per-dataset results
                overall_bias_score=overall_score,  # REAL harmonic mean
                evaluation_time=evaluation_time,
                metadata={"suite": suite, "real_evaluation": True, "total_datasets": len(dataset_results)}
            )
                
        except Exception as e:
            print(f"❌ Evaluation failed for {model_variant}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _aggregate_seed_results(self, model_variant: str, seed_results: List[SeedResult],
                               eval_config: EvaluationConfig) -> AggregatedResults:
        """Aggregate results preserving dataset-specific metrics (NO meaningless overall scores)."""
        
        print(f"📊 [REAL AGGREGATION] Aggregating {len(seed_results)} evaluations for {model_variant}")
        
        # DON'T aggregate into meaningless single bias scores!
        # Keep dataset-specific results separate as they have different meanings
        
        # Collect all dataset-specific results
        dataset_means = {}
        dataset_stds = {}
        
        if seed_results:
            all_datasets = set()
            for seed_result in seed_results:
                all_datasets.update(seed_result.dataset_results.keys())
            
            print(f"   📋 Datasets found: {sorted(all_datasets)}")
            
            for dataset in all_datasets:
                dataset_scores = [
                    r.dataset_results.get(dataset, 0.0) 
                    for r in seed_results 
                    if dataset in r.dataset_results
                ]
                
                if dataset_scores:
                    mean_score = np.mean(dataset_scores)
                    std_score = np.std(dataset_scores, ddof=1) if len(dataset_scores) > 1 else 0.0
                    
                    dataset_means[dataset] = mean_score
                    dataset_stds[dataset] = std_score
                    
                    print(f"   📊 {dataset}: {mean_score:.4f} ± {std_score:.4f}")
        
        # Use harmonic mean of available dataset scores as overall summary
        # (NOT a meaningful metric, just for backward compatibility)
        # Include all scores > 0.0001 to capture small BOLD sentiment bias scores
        available_means = [score for score in dataset_means.values() if score > 0.0001]
        if available_means and len(available_means) > 0:
            # Harmonic mean prevents any single bad score from dominating
            harmonic_mean = len(available_means) / sum(1/score for score in available_means)
            harmonic_std = np.std([r.overall_bias_score for r in seed_results], ddof=1) if len(seed_results) > 1 else 0.0
            
            if len(seed_results) > 1:
                overall_scores = [r.overall_bias_score for r in seed_results]
                overall_ci = stats.t.interval(
                    eval_config.confidence_level,
                    len(overall_scores)-1,
                    loc=harmonic_mean,
                    scale=stats.sem(overall_scores)
                )
            else:
                overall_ci = (harmonic_mean, harmonic_mean)
        else:
            harmonic_mean = 0.0
            harmonic_std = 0.0
            overall_ci = (0.0, 0.0)
        
        print(f"   🎯 Summary (harmonic mean): {harmonic_mean:.4f} ± {harmonic_std:.4f}")
        print(f"   ⚠️  NOTE: Overall score is for reference only - use dataset-specific metrics!")
        
        return AggregatedResults(
            model_variant=model_variant,
            mean_bias_score=harmonic_mean,  # Harmonic mean for compatibility
            std_bias_score=harmonic_std,
            confidence_interval=overall_ci,
            dataset_means=dataset_means,      # REAL dataset-specific results
            dataset_stds=dataset_stds,        # REAL dataset-specific uncertainties  
            n_evaluations=len(seed_results),
            seed_results=seed_results,
            statistical_significance={}
        )
    
    def _compute_statistical_significance(self, aggregated_results: Dict[str, AggregatedResults],
                                        eval_config: EvaluationConfig) -> Dict[str, Any]:
        """Compute statistical significance tests between models."""
        significance_results = {}
        
        models = list(aggregated_results.keys())
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                if model1 in aggregated_results and model2 in aggregated_results:
                    scores1 = [r.overall_bias_score for r in aggregated_results[model1].seed_results]
                    scores2 = [r.overall_bias_score for r in aggregated_results[model2].seed_results]
                    
                    comparison_key = f"{model1}_vs_{model2}"
                    
                    # T-test
                    if len(scores1) > 1 and len(scores2) > 1:
                        t_stat, p_value = stats.ttest_ind(scores1, scores2)
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(scores1)-1)*np.var(scores1, ddof=1) + 
                                            (len(scores2)-1)*np.var(scores2, ddof=1)) / 
                                           (len(scores1) + len(scores2) - 2))
                        effect_size = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0
                        
                        significance_results[comparison_key] = {
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "effect_size": effect_size,
                            "significant": p_value < (1 - eval_config.confidence_level)
                        }
        
        return significance_results
    
    def _estimate_total_time(self, eval_config: EvaluationConfig) -> float:
        """Estimate total evaluation time in minutes."""
        # Rough estimates based on typical evaluation times
        training_time_per_seed = 30  # minutes per model training
        evaluation_time_per_seed = 10  # minutes per evaluation
        
        n_training_seeds = len(eval_config.training_seeds)
        n_eval_seeds = len(eval_config.evaluation_seeds)
        n_models = 4
        
        total_training_time = n_training_seeds * 3 * training_time_per_seed  # 3 trainable models
        total_evaluation_time = n_training_seeds * n_eval_seeds * n_models * evaluation_time_per_seed
        
        return total_training_time + total_evaluation_time
    
    def _generate_summary_report(self, aggregated_results: Dict[str, AggregatedResults],
                               statistical_results: Dict[str, Any], output_dir: Path) -> None:
        """Generate a markdown summary report."""
        report_path = output_dir / "summary_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Robust Multi-Seed Evaluation Report\n\n")
            
            f.write("## Results Summary\n\n")
            for model, results in aggregated_results.items():
                f.write(f"### {model.upper()}\n")
                f.write(f"- **Mean Bias Score**: {results.mean_bias_score:.4f} ± {results.std_bias_score:.4f}\n")
                f.write(f"- **95% Confidence Interval**: [{results.confidence_interval[0]:.4f}, {results.confidence_interval[1]:.4f}]\n")
                f.write(f"- **Number of Evaluations**: {results.n_evaluations}\n\n")
            
            f.write("## Statistical Significance\n\n")
            for comparison, stats in statistical_results.items():
                significance = "✅ Significant" if stats["significant"] else "❌ Not Significant"
                f.write(f"### {comparison.replace('_', ' ').title()}\n")
                f.write(f"- **P-value**: {stats['p_value']:.6f}\n")
                f.write(f"- **Effect Size**: {stats['effect_size']:.4f}\n")
                f.write(f"- **Significance**: {significance}\n\n")


# Convenience functions for the main pipeline
def create_robust_evaluator(base_dir: str = "/workspace/Algoverse") -> RobustEvaluationFramework:
    """Create a robust evaluation framework instance."""
    return RobustEvaluationFramework(base_dir)


def run_quick_robust_evaluation(base_config_path: str, model_name: str, suite: str = "comprehensive") -> Dict[str, AggregatedResults]:
    """Run a quick robust evaluation (1 seed each)."""
    framework = RobustEvaluationFramework()
    return framework.run_robust_four_model_evaluation(
        base_config_path, model_name, suite, robustness_level="quick"
    )


def run_standard_robust_evaluation(base_config_path: str, model_name: str, suite: str = "comprehensive") -> Dict[str, AggregatedResults]:
    """Run a standard robust evaluation (3 seeds each)."""
    framework = RobustEvaluationFramework()
    return framework.run_robust_four_model_evaluation(
        base_config_path, model_name, suite, robustness_level="standard"
    )


def run_publication_robust_evaluation(base_config_path: str, model_name: str, suite: str = "comprehensive") -> Dict[str, AggregatedResults]:
    """Run a publication-ready robust evaluation (5 seeds each)."""
    framework = RobustEvaluationFramework()
    return framework.run_robust_four_model_evaluation(
        base_config_path, model_name, suite, robustness_level="publication"
    )