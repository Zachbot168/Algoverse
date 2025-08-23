#!/usr/bin/env python3
"""
GPU Optimization Utilities for FIRM Pipeline

Automatically detects GPU capabilities and optimizes batch sizes and memory usage
for maximum efficiency during model training and evaluation.
"""

import torch
import subprocess
from typing import Dict, Tuple, Optional
import json

class GPUOptimizer:
    """Automatically optimize GPU settings for FIRM pipeline."""
    
    def __init__(self):
        """Initialize GPU optimizer with current hardware detection."""
        self.gpu_info = self._detect_gpu()
        self.optimization_settings = self._calculate_optimal_settings()
    
    def _detect_gpu(self) -> Dict[str, any]:
        """Detect current GPU capabilities."""
        gpu_info = {
            'available': torch.cuda.is_available(),
            'device_count': 0,
            'total_memory': 0,
            'name': 'CPU',
            'compute_capability': None
        }
        
        if torch.cuda.is_available():
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['name'] = torch.cuda.get_device_name(0)
            gpu_info['compute_capability'] = torch.cuda.get_device_capability(0)
            
            # Get memory info
            memory_info = torch.cuda.get_device_properties(0)
            gpu_info['total_memory'] = memory_info.total_memory // (1024**3)  # GB
            
            # Try to get more detailed info via nvidia-smi
            try:
                result = subprocess.check_output([
                    'nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'
                ], text=True)
                gpu_info['total_memory'] = int(result.strip()) // 1024  # Convert MB to GB
            except:
                pass
        
        return gpu_info
    
    def _calculate_optimal_settings(self) -> Dict[str, any]:
        """Calculate optimal batch sizes and memory settings."""
        settings = {
            'evaluation_batch_size': 1,
            'training_batch_size': 1,
            'gradient_accumulation_steps': 4,
            'use_fp16': True,
            'use_gradient_checkpointing': False,
            'max_workers': 4
        }
        
        if not self.gpu_info['available']:
            return settings
        
        total_memory = self.gpu_info['total_memory']
        
        # Optimize based on available VRAM
        if total_memory >= 80:  # H100, A100 80GB, H200
            settings.update({
                'evaluation_batch_size': 16,
                'training_batch_size': 4,
                'gradient_accumulation_steps': 2,
                'use_gradient_checkpointing': False,
                'max_workers': 8
            })
        elif total_memory >= 40:  # A100 40GB, A6000
            settings.update({
                'evaluation_batch_size': 12,
                'training_batch_size': 3,
                'gradient_accumulation_steps': 4,
                'use_gradient_checkpointing': False,
                'max_workers': 6
            })
        elif total_memory >= 24:  # RTX 4090, RTX A6000
            settings.update({
                'evaluation_batch_size': 8,
                'training_batch_size': 2,
                'gradient_accumulation_steps': 4,
                'use_gradient_checkpointing': True,
                'max_workers': 4
            })
        elif total_memory >= 12:  # RTX 4070 Ti, RTX 3080 Ti
            settings.update({
                'evaluation_batch_size': 4,
                'training_batch_size': 1,
                'gradient_accumulation_steps': 8,
                'use_gradient_checkpointing': True,
                'max_workers': 4
            })
        elif total_memory >= 8:  # RTX 3070, RTX 4060 Ti
            settings.update({
                'evaluation_batch_size': 2,
                'training_batch_size': 1,
                'gradient_accumulation_steps': 16,
                'use_gradient_checkpointing': True,
                'max_workers': 2
            })
        
        return settings
    
    def get_optimization_report(self) -> str:
        """Generate a detailed optimization report."""
        report = f"""
🚀 GPU OPTIMIZATION REPORT
{'='*50}

Hardware Detection:
  GPU Available: {'✅' if self.gpu_info['available'] else '❌'}
  Device Count: {self.gpu_info['device_count']}
  GPU Name: {self.gpu_info['name']}
  Total VRAM: {self.gpu_info['total_memory']}GB
  Compute Capability: {self.gpu_info['compute_capability']}

Optimized Settings:
  Evaluation Batch Size: {self.optimization_settings['evaluation_batch_size']}
  Training Batch Size: {self.optimization_settings['training_batch_size']}
  Gradient Accumulation: {self.optimization_settings['gradient_accumulation_steps']}
  Use FP16: {'✅' if self.optimization_settings['use_fp16'] else '❌'}
  Gradient Checkpointing: {'✅' if self.optimization_settings['use_gradient_checkpointing'] else '❌'}
  Max Workers: {self.optimization_settings['max_workers']}

Performance Recommendations:
"""
        
        if self.gpu_info['total_memory'] >= 80:
            report += "  🚀 High-end GPU detected - Maximum performance settings applied\n"
            report += "  💡 Consider using multiple models in parallel\n"
        elif self.gpu_info['total_memory'] >= 24:
            report += "  ✅ Good GPU detected - Balanced performance settings applied\n"
            report += "  💡 Should handle 3B models efficiently\n"
        elif self.gpu_info['total_memory'] >= 8:
            report += "  ⚠️  Mid-range GPU detected - Memory-optimized settings applied\n"
            report += "  💡 Consider using smaller batch sizes if OOM occurs\n"
        else:
            report += "  ❌ Limited VRAM detected - Conservative settings applied\n"
            report += "  💡 Consider using CPU evaluation for large models\n"
        
        report += f"\n{'='*50}"
        return report
    
    def optimize_config(self, config: Dict) -> Dict:
        """Automatically optimize a model configuration."""
        optimized_config = config.copy()
        
        # Update evaluation settings
        if 'evaluation' in optimized_config:
            optimized_config['evaluation']['batch_size'] = self.optimization_settings['evaluation_batch_size']
        
        # Update training settings
        if 'interventions' in optimized_config and 'pinpoint_tuning' in optimized_config['interventions']:
            training_config = optimized_config['interventions']['pinpoint_tuning']['training']
            training_config['per_device_train_batch_size'] = self.optimization_settings['training_batch_size']
            training_config['gradient_accumulation_steps'] = self.optimization_settings['gradient_accumulation_steps']
            training_config['dataloader_num_workers'] = self.optimization_settings['max_workers']
            
            if self.optimization_settings['use_gradient_checkpointing']:
                training_config['gradient_checkpointing'] = True
        
        # Update model precision
        if 'model' in optimized_config:
            if self.optimization_settings['use_fp16']:
                optimized_config['model']['torch_dtype'] = 'float16'
        
        return optimized_config
    
    def save_optimization_report(self, filepath: str):
        """Save optimization report to file."""
        with open(filepath, 'w') as f:
            f.write(self.get_optimization_report())
        print(f"📊 Optimization report saved to: {filepath}")


def optimize_model_config(config_path: str, output_path: Optional[str] = None) -> str:
    """
    Optimize a model configuration file for current GPU.
    
    Args:
        config_path: Path to model configuration YAML file
        output_path: Path to save optimized config (optional)
        
    Returns:
        Path to optimized configuration file
    """
    import yaml
    
    # Load original config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Optimize config
    optimizer = GPUOptimizer()
    optimized_config = optimizer.optimize_config(config)
    
    # Save optimized config
    if output_path is None:
        output_path = config_path.replace('.yaml', '_optimized.yaml')
    
    with open(output_path, 'w') as f:
        yaml.dump(optimized_config, f, indent=2)
    
    print(f"✅ Optimized configuration saved to: {output_path}")
    print(optimizer.get_optimization_report())
    
    return output_path


if __name__ == "__main__":
    # Command line usage
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Optimize FIRM pipeline GPU settings")
    parser.add_argument("--config", help="Model config file to optimize")
    parser.add_argument("--output", help="Output path for optimized config")
    parser.add_argument("--report", help="Save optimization report to file")
    
    args = parser.parse_args()
    
    optimizer = GPUOptimizer()
    
    if args.config:
        optimize_model_config(args.config, args.output)
    else:
        print(optimizer.get_optimization_report())
    
    if args.report:
        optimizer.save_optimization_report(args.report)