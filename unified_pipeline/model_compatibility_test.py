#!/usr/bin/env python3
"""
Model Compatibility Test Suite

Tests the unified pipeline with all supported models to ensure compatibility
and proper configuration. This script validates that each model can be loaded
and processed through the pipeline without errors.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.model_adapter import create_model_adapter, UniversalModelAdapter

warnings.filterwarnings('ignore')


class ModelCompatibilityTester:
    """Tests model compatibility with the unified pipeline."""
    
    def __init__(self, models_dir: str = "../models"):
        """Initialize the compatibility tester."""
        self.models_dir = Path(models_dir)
        
        # Load model registry
        registry_path = self.models_dir / "model_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
            self.models = registry_data['model_registry']['models']
        else:
            print("⚠️  Model registry not found, using default test models")
            self.models = {
                "gpt2": {"huggingface_id": "gpt2", "model_type": "decoder"},
                "bert-base-uncased": {"huggingface_id": "bert-base-uncased", "model_type": "encoder"}
            }
        
        print(f"Initialized compatibility tester for {len(self.models)} models")
    
    def test_model_loading(self, model_name: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Test if a model can be loaded with the universal adapter.
        
        Args:
            model_name: Name of model to test
            
        Returns:
            Tuple of (success, error_message, model_info)
        """
        print(f"\n🧪 Testing model loading: {model_name}")
        
        if model_name not in self.models:
            return False, f"Model {model_name} not found in registry", {}
        
        model_info = self.models[model_name]
        huggingface_id = model_info['huggingface_id']
        
        try:
            # Check if model exists locally
            local_model_path = self.models_dir / model_name
            if local_model_path.exists():
                print(f"   Using local model: {local_model_path}")
                adapter = create_model_adapter(str(local_model_path), device="cpu")
            else:
                print(f"   Loading from HuggingFace: {huggingface_id}")
                adapter = create_model_adapter(huggingface_id, device="cpu")
            
            # Test compatibility
            is_compatible, issues = adapter.is_compatible_with_pipeline()
            
            info = {
                "architecture": adapter.arch_info.architecture,
                "model_type": adapter.arch_info.model_type,
                "num_layers": adapter.arch_info.num_layers,
                "hidden_size": adapter.arch_info.hidden_size,
                "num_heads": adapter.arch_info.num_heads,
                "compatible": is_compatible,
                "issues": issues,
                "lora_targets": adapter.get_lora_target_modules()
            }
            
            print(f"   ✅ Successfully loaded {adapter.arch_info.architecture} model")
            print(f"   Architecture: {info['architecture']} ({info['model_type']})")
            print(f"   Layers: {info['num_layers']}, Hidden: {info['hidden_size']}, Heads: {info['num_heads']}")
            
            if not is_compatible:
                print(f"   ⚠️  Compatibility issues: {issues}")
            
            return True, "", info
            
        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {str(e)}"
            print(f"   ❌ {error_msg}")
            return False, error_msg, {}
    
    def test_model_forward_pass(self, adapter: UniversalModelAdapter) -> Tuple[bool, str]:
        """Test basic forward pass functionality."""
        try:
            # Test with simple inputs
            test_texts = ["This is a test sentence.", "Another test sentence for batch processing."]
            
            # Prepare inputs
            inputs = adapter.prepare_inputs(test_texts, max_length=64)
            
            # Run forward pass
            outputs = adapter.forward_pass(inputs, output_hidden_states=True)
            
            # Check outputs
            if 'last_hidden_state' not in outputs:
                return False, "Missing last_hidden_state in outputs"
            
            hidden_states = outputs['last_hidden_state']
            expected_shape = (len(test_texts), inputs['input_ids'].size(1), adapter.arch_info.hidden_size)
            
            if hidden_states.shape != expected_shape:
                return False, f"Unexpected hidden state shape: {hidden_states.shape} vs {expected_shape}"
            
            print(f"   ✅ Forward pass successful, output shape: {hidden_states.shape}")
            return True, ""
            
        except Exception as e:
            return False, f"Forward pass failed: {str(e)}"
    
    def test_model_generation(self, adapter: UniversalModelAdapter) -> Tuple[bool, str]:
        """Test text generation for decoder models."""
        if adapter.arch_info.model_type != 'decoder':
            return True, "Skipped (encoder model)"
        
        try:
            generated_text = adapter.generate_text(
                "The future of AI is",
                max_new_tokens=10,
                temperature=0.8
            )
            
            if not generated_text or len(generated_text.strip()) == 0:
                return False, "Generated empty text"
            
            print(f"   ✅ Generation successful: '{generated_text.strip()}'")
            return True, ""
            
        except Exception as e:
            return False, f"Generation failed: {str(e)}"
    
    def test_hook_registration(self, adapter: UniversalModelAdapter) -> Tuple[bool, str]:
        """Test hook registration for different layers."""
        try:
            # Test layer hook
            def dummy_hook(module, input, output):
                pass
            
            hook_handle = adapter.register_layer_hook(0, dummy_hook)
            hook_handle.remove()
            
            # Test attention hook
            try:
                attn_hook = adapter.register_attention_hook(0, dummy_hook)
                attn_hook.remove()
                print(f"   ✅ Attention hooks working")
            except Exception as e:
                print(f"   ⚠️  Attention hooks failed: {e}")
            
            # Test MLP hook
            try:
                mlp_hook = adapter.register_mlp_hook(0, dummy_hook)
                mlp_hook.remove()
                print(f"   ✅ MLP hooks working")
            except Exception as e:
                print(f"   ⚠️  MLP hooks failed: {e}")
            
            return True, ""
            
        except Exception as e:
            return False, f"Hook registration failed: {str(e)}"
    
    def test_single_model(self, model_name: str) -> Dict[str, Any]:
        """Run complete test suite for a single model."""
        results = {
            "model_name": model_name,
            "loading": {"success": False, "error": "", "info": {}},
            "forward_pass": {"success": False, "error": ""},
            "generation": {"success": False, "error": ""},
            "hooks": {"success": False, "error": ""},
            "overall_success": False
        }
        
        # Test 1: Model loading
        success, error, info = self.test_model_loading(model_name)
        results["loading"] = {"success": success, "error": error, "info": info}
        
        if not success:
            return results
        
        # Create adapter for further tests
        try:
            model_info = self.models[model_name]
            huggingface_id = model_info['huggingface_id']
            
            local_model_path = self.models_dir / model_name
            if local_model_path.exists():
                adapter = create_model_adapter(str(local_model_path), device="cpu")
            else:
                adapter = create_model_adapter(huggingface_id, device="cpu")
        except:
            return results
        
        # Test 2: Forward pass
        success, error = self.test_model_forward_pass(adapter)
        results["forward_pass"] = {"success": success, "error": error}
        
        # Test 3: Generation (for decoder models)
        success, error = self.test_model_generation(adapter)
        results["generation"] = {"success": success, "error": error}
        
        # Test 4: Hook registration
        success, error = self.test_hook_registration(adapter)
        results["hooks"] = {"success": success, "error": error}
        
        # Overall success
        results["overall_success"] = (
            results["loading"]["success"] and
            results["forward_pass"]["success"] and
            results["generation"]["success"] and
            results["hooks"]["success"]
        )
        
        return results
    
    def test_all_models(self, models_to_test: List[str] = None) -> Dict[str, Any]:
        """Test all models or a specific subset."""
        if models_to_test is None:
            models_to_test = list(self.models.keys())
        
        print(f"\n🚀 Testing {len(models_to_test)} models for pipeline compatibility")
        print("=" * 70)
        
        all_results = {}
        successful_models = []
        failed_models = []
        
        for model_name in models_to_test:
            print(f"\n{'='*20} {model_name} {'='*20}")
            
            results = self.test_single_model(model_name)
            all_results[model_name] = results
            
            if results["overall_success"]:
                successful_models.append(model_name)
                print(f"✅ {model_name}: PASSED")
            else:
                failed_models.append(model_name)
                print(f"❌ {model_name}: FAILED")
                
                # Print failure details
                for test_name, test_result in results.items():
                    if test_name in ["loading", "forward_pass", "generation", "hooks"]:
                        if not test_result["success"] and test_result["error"]:
                            print(f"   {test_name}: {test_result['error']}")
        
        # Summary
        print(f"\n{'='*70}")
        print(f"📊 COMPATIBILITY TEST SUMMARY")
        print(f"{'='*70}")
        print(f"✅ Successful: {len(successful_models)}/{len(models_to_test)}")
        print(f"❌ Failed: {len(failed_models)}/{len(models_to_test)}")
        
        if successful_models:
            print(f"\n✅ Compatible models: {', '.join(successful_models)}")
        
        if failed_models:
            print(f"\n❌ Incompatible models: {', '.join(failed_models)}")
        
        # Save results
        results_file = Path("model_compatibility_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        return all_results
    
    def generate_compatibility_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed compatibility report."""
        report_lines = []
        report_lines.append("# Model Compatibility Report")
        report_lines.append("")
        
        # Summary table
        report_lines.append("## Summary")
        report_lines.append("")
        report_lines.append("| Model | Architecture | Loading | Forward | Generation | Hooks | Overall |")
        report_lines.append("|-------|-------------|---------|---------|------------|-------|---------|")
        
        for model_name, result in results.items():
            arch = result["loading"]["info"].get("architecture", "unknown")
            loading = "✅" if result["loading"]["success"] else "❌" 
            forward = "✅" if result["forward_pass"]["success"] else "❌"
            generation = "✅" if result["generation"]["success"] else "❌"
            hooks = "✅" if result["hooks"]["success"] else "❌"
            overall = "✅" if result["overall_success"] else "❌"
            
            report_lines.append(f"| {model_name} | {arch} | {loading} | {forward} | {generation} | {hooks} | {overall} |")
        
        # Detailed results
        report_lines.append("")
        report_lines.append("## Detailed Results")
        report_lines.append("")
        
        for model_name, result in results.items():
            report_lines.append(f"### {model_name}")
            report_lines.append("")
            
            info = result["loading"]["info"]
            if info:
                report_lines.append(f"- **Architecture**: {info.get('architecture', 'unknown')}")
                report_lines.append(f"- **Model Type**: {info.get('model_type', 'unknown')}")
                report_lines.append(f"- **Layers**: {info.get('num_layers', 'unknown')}")
                report_lines.append(f"- **Hidden Size**: {info.get('hidden_size', 'unknown')}")
                report_lines.append(f"- **Attention Heads**: {info.get('num_heads', 'unknown')}")
                
                if info.get('issues'):
                    report_lines.append(f"- **Issues**: {', '.join(info['issues'])}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


def main():
    """Main entry point for compatibility testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model compatibility with unified pipeline")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--models-dir", default="../models", help="Models directory")
    parser.add_argument("--report", action="store_true", help="Generate compatibility report")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ModelCompatibilityTester(args.models_dir)
    
    # Run tests
    results = tester.test_all_models(args.models)
    
    # Generate report if requested
    if args.report:
        report = tester.generate_compatibility_report(results)
        report_file = Path("model_compatibility_report.md")
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📄 Compatibility report saved to: {report_file}")


if __name__ == "__main__":
    main()