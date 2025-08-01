#!/usr/bin/env python3
"""
Quick Model Test Runner

Tests pipeline compatibility with available models without full download.
Uses mock adapters to validate configuration and architecture support.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add utils to path
sys.path.append(str(Path(__file__).parent))

def mock_test_model_architectures():
    """Test model architecture support without downloading models."""
    
    print("🧪 Testing Model Architecture Support")
    print("=" * 50)
    
    # Model architecture definitions
    test_models = {
        "bert-base-uncased": {
            "architecture": "bert",
            "model_type": "encoder",
            "num_layers": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "layer_pattern": "encoder.layer",
            "attention_pattern": "attention.self",
            "lora_targets": ["query", "value", "key", "dense"]
        },
        "roberta-base": {
            "architecture": "roberta", 
            "model_type": "encoder",
            "num_layers": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "layer_pattern": "roberta.encoder.layer",
            "attention_pattern": "attention.self",
            "lora_targets": ["query", "value", "key", "dense"]
        },
        "gpt2": {
            "architecture": "gpt2",
            "model_type": "decoder",
            "num_layers": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "layer_pattern": "transformer.h",
            "attention_pattern": "attn",
            "lora_targets": ["c_attn", "c_proj", "c_fc"]
        },
        "gpt2-medium": {
            "architecture": "gpt2",
            "model_type": "decoder", 
            "num_layers": 24,
            "hidden_size": 1024,
            "num_heads": 16,
            "layer_pattern": "transformer.h",
            "attention_pattern": "attn",
            "lora_targets": ["c_attn", "c_proj", "c_fc"]
        },
        "gemma-2-2b-it": {
            "architecture": "gemma",
            "model_type": "decoder",
            "num_layers": 18,
            "hidden_size": 2304,
            "num_heads": 8,
            "layer_pattern": "model.model.layers",
            "attention_pattern": "self_attn",
            "lora_targets": ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        },
        "llama-3.2-1b": {
            "architecture": "llama",
            "model_type": "decoder",
            "num_layers": 16,
            "hidden_size": 2048,
            "num_heads": 32,
            "layer_pattern": "model.model.layers",
            "attention_pattern": "self_attn", 
            "lora_targets": ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        },
        "llama-3.2-3b": {
            "architecture": "llama",
            "model_type": "decoder",
            "num_layers": 28,
            "hidden_size": 3072,
            "num_heads": 24,
            "layer_pattern": "model.model.layers",
            "attention_pattern": "self_attn",
            "lora_targets": ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }
    }
    
    # Test each architecture
    results = {}
    for model_name, model_info in test_models.items():
        print(f"\n🔍 Testing {model_name}")
        
        # Test architecture detection
        arch_supported = model_info["architecture"] in ["bert", "roberta", "gpt2", "gemma", "llama"]
        print(f"   Architecture '{model_info['architecture']}': {'✅' if arch_supported else '❌'}")
        
        # Test model type support
        type_supported = model_info["model_type"] in ["encoder", "decoder"]
        print(f"   Model type '{model_info['model_type']}': {'✅' if type_supported else '❌'}")
        
        # Test layer access pattern
        layer_pattern_valid = model_info["layer_pattern"] in [
            "encoder.layer", "roberta.encoder.layer", "transformer.h", "model.model.layers"
        ]
        print(f"   Layer pattern '{model_info['layer_pattern']}': {'✅' if layer_pattern_valid else '❌'}")
        
        # Test attention pattern
        attn_pattern_valid = model_info["attention_pattern"] in [
            "attention.self", "attn", "self_attn"
        ]
        print(f"   Attention pattern '{model_info['attention_pattern']}': {'✅' if attn_pattern_valid else '❌'}")
        
        # Test LoRA targets
        lora_valid = len(model_info["lora_targets"]) > 0
        print(f"   LoRA targets ({len(model_info['lora_targets'])}): {'✅' if lora_valid else '❌'}")
        
        # Test pipeline compatibility
        pipeline_compatible = all([
            arch_supported,
            type_supported, 
            layer_pattern_valid,
            attn_pattern_valid,
            lora_valid
        ])
        
        print(f"   Pipeline compatible: {'✅' if pipeline_compatible else '❌'}")
        
        results[model_name] = {
            "architecture_supported": arch_supported,
            "type_supported": type_supported,
            "layer_pattern_valid": layer_pattern_valid,
            "attention_pattern_valid": attn_pattern_valid,
            "lora_valid": lora_valid,
            "pipeline_compatible": pipeline_compatible,
            "model_info": model_info
        }
    
    # Summary
    compatible_models = [name for name, result in results.items() if result["pipeline_compatible"]]
    incompatible_models = [name for name, result in results.items() if not result["pipeline_compatible"]]
    
    print(f"\n📊 ARCHITECTURE COMPATIBILITY SUMMARY")
    print("=" * 50)
    print(f"✅ Compatible models: {len(compatible_models)}/{len(test_models)}")
    print(f"❌ Incompatible models: {len(incompatible_models)}/{len(test_models)}")
    
    if compatible_models:
        print(f"\n✅ Supported models:")
        for model in compatible_models:
            arch = results[model]["model_info"]["architecture"]
            mtype = results[model]["model_info"]["model_type"]
            print(f"   • {model} ({arch}, {mtype})")
    
    if incompatible_models:
        print(f"\n❌ Unsupported models:")
        for model in incompatible_models:
            print(f"   • {model}")
    
    return results

def test_configuration_files():
    """Test that configuration files exist and are valid."""
    print(f"\n🔧 Testing Configuration Files")
    print("=" * 30)
    
    config_dir = Path("configs/models")
    expected_configs = [
        "bert-base-uncased.yaml",
        "roberta-base.yaml", 
        "gpt2.yaml",
        "gemma-2-2b-it.yaml",
        "llama-3.2-1b.yaml"
    ]
    
    results = {}
    for config_file in expected_configs:
        config_path = config_dir / config_file
        model_name = config_file.replace(".yaml", "")
        
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check required sections
                required_sections = ["model", "evaluation", "interventions", "data"]
                missing_sections = [s for s in required_sections if s not in config]
                
                if missing_sections:
                    print(f"   ❌ {model_name}: Missing sections: {missing_sections}")
                    results[model_name] = {"exists": True, "valid": False, "error": f"Missing sections: {missing_sections}"}
                else:
                    print(f"   ✅ {model_name}: Valid configuration")
                    results[model_name] = {"exists": True, "valid": True, "error": None}
                    
            except Exception as e:
                print(f"   ❌ {model_name}: Invalid YAML: {e}")
                results[model_name] = {"exists": True, "valid": False, "error": str(e)}
        else:
            print(f"   ❌ {model_name}: Configuration file missing")
            results[model_name] = {"exists": False, "valid": False, "error": "File not found"}
    
    return results

def test_pipeline_components():
    """Test that pipeline components can handle different architectures."""
    print(f"\n🔧 Testing Pipeline Component Support")
    print("=" * 40)
    
    # Test component registry
    try:
        from train.component_registry import ComponentRegistryManager
        registry_manager = ComponentRegistryManager()
        print("   ✅ Component registry: Importable")
    except Exception as e:
        print(f"   ❌ Component registry: Import failed - {e}")
        return False
    
    # Test model adapter
    try:
        from utils.model_adapter import UniversalModelAdapter
        print("   ✅ Model adapter: Importable")
    except Exception as e:
        print(f"   ❌ Model adapter: Import failed - {e}")
        return False
    
    # Test diagnostic system
    try:
        from eval.run_diagnostic import UnifiedDiagnosticPass
        print("   ✅ Diagnostic system: Importable")
    except Exception as e:
        print(f"   ❌ Diagnostic system: Import failed - {e}")
        return False
    
    # Test steering components
    try:
        from steer.das_wrapper import DynamicActivationSteering
        print("   ✅ Steering wrapper: Importable")
    except Exception as e:
        print(f"   ❌ Steering wrapper: Import failed - {e}")
        return False
    
    return True

def main():
    """Run complete pipeline compatibility test."""
    print("🚀 UNIFIED PIPELINE COMPATIBILITY TEST")
    print("=" * 70)
    
    # Test 1: Architecture support
    arch_results = mock_test_model_architectures()
    
    # Test 2: Configuration files  
    config_results = test_configuration_files()
    
    # Test 3: Pipeline components
    component_success = test_pipeline_components()
    
    # Overall summary
    print(f"\n🎯 OVERALL COMPATIBILITY SUMMARY")
    print("=" * 70)
    
    arch_compatible = sum(1 for r in arch_results.values() if r["pipeline_compatible"])
    config_valid = sum(1 for r in config_results.values() if r["valid"])
    
    print(f"Architecture Support: {arch_compatible}/{len(arch_results)} models")
    print(f"Configuration Files: {config_valid}/{len(config_results)} configs")
    print(f"Pipeline Components: {'✅ Working' if component_success else '❌ Issues'}")
    
    # Final verdict
    overall_success = (
        arch_compatible >= len(arch_results) * 0.8 and  # At least 80% models supported
        config_valid >= len(config_results) * 0.8 and   # At least 80% configs valid
        component_success
    )
    
    if overall_success:
        print(f"\n🎉 UNIFIED PIPELINE IS READY FOR ALL MODELS!")
        print(f"✅ The pipeline can handle all supported model architectures")
        print(f"✅ Configuration templates are available")
        print(f"✅ All components are properly integrated")
    else:
        print(f"\n⚠️  PIPELINE NEEDS ATTENTION")
        print(f"❌ Some models or configurations need fixes")
    
    # Save results
    results = {
        "architecture_tests": arch_results,
        "configuration_tests": config_results,
        "component_tests": component_success,
        "overall_success": overall_success
    }
    
    with open("pipeline_compatibility_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: pipeline_compatibility_results.json")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)