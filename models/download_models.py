#!/usr/bin/env python3
"""
Model Download Manager

Downloads and manages open-source models for bias evaluation and testing.
Ensures models are stored locally and never committed to the repository.

Usage:
    python download_models.py --models bert-base-uncased gpt2
    python download_models.py --collection small_models
    python download_models.py --all
    python download_models.py --list
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

try:
    from huggingface_hub import snapshot_download, login, HfApi
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    import torch
except ImportError as e:
    print(f"Error: Required packages not installed. Please run:")
    print(f"pip install transformers huggingface_hub torch")
    sys.exit(1)

warnings.filterwarnings('ignore')


class ModelDownloadManager:
    """Manages downloading and organizing open-source models."""
    
    def __init__(self, registry_path: str = "model_registry.json", 
                 models_dir: str = "./models"):
        """
        Initialize the model download manager.
        
        Args:
            registry_path: Path to model registry JSON file
            models_dir: Directory to store downloaded models
        """
        self.models_dir = Path(models_dir)
        self.cache_dir = self.models_dir / "cache"
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load model registry
        registry_file = self.models_dir / registry_path
        if not registry_file.exists():
            raise FileNotFoundError(f"Model registry not found: {registry_file}")
        
        with open(registry_file, 'r') as f:
            self.registry_data = json.load(f)
        
        self.models = self.registry_data['model_registry']['models']
        self.collections = self.registry_data['model_registry']['model_collections']
        self.settings = self.registry_data['download_settings']
        self.storage_req = self.registry_data['storage_requirements']
        
        # Initialize HuggingFace API
        self.hf_api = HfApi()
        self.authenticated = False
        
        print(f"Initialized ModelDownloadManager")
        print(f"Models directory: {self.models_dir}")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Registry contains {len(self.models)} models")
        
    def authenticate_huggingface(self, token: Optional[str] = None) -> bool:
        """
        Authenticate with Hugging Face Hub.
        
        Args:
            token: HuggingFace token (optional, will prompt if not provided)
            
        Returns:
            True if authentication successful
        """
        if self.authenticated:
            return True
            
        try:
            if token:
                login(token=token)
            else:
                # Try to use existing token or prompt
                login()
            
            # Test authentication
            user_info = self.hf_api.whoami()
            print(f"✅ Authenticated as: {user_info['name']}")
            self.authenticated = True
            return True
            
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            print("Some models require authentication. Use --token <your_token> or set HF_TOKEN environment variable")
            return False
    
    def check_storage_space(self, required_gb: float) -> bool:
        """
        Check if there's enough storage space.
        
        Args:
            required_gb: Required space in GB
            
        Returns:
            True if sufficient space available
        """
        try:
            # Get available space
            stat = shutil.disk_usage(self.models_dir)
            available_gb = stat.free / (1024**3)
            
            print(f"Available space: {available_gb:.1f} GB")
            print(f"Required space: {required_gb:.1f} GB")
            
            if available_gb < required_gb:
                print(f"❌ Insufficient storage space!")
                print(f"Need at least {required_gb:.1f} GB, but only {available_gb:.1f} GB available")
                return False
            
            if available_gb < self.storage_req['warning_threshold_gb']:
                print(f"⚠️  Warning: Low disk space ({available_gb:.1f} GB remaining)")
                
            return True
            
        except Exception as e:
            print(f"Warning: Could not check disk space: {e}")
            return True  # Proceed anyway
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model information from registry."""
        return self.models.get(model_name)
    
    def list_models(self, filter_by: Optional[str] = None) -> None:
        """
        List available models.
        
        Args:
            filter_by: Filter by collection name (optional)
        """
        print(f"\n📋 Available Models:")
        print("=" * 80)
        
        models_to_show = []
        
        if filter_by and filter_by in self.collections:
            models_to_show = self.collections[filter_by]
            print(f"Collection: {filter_by}")
        else:
            models_to_show = list(self.models.keys())
        
        for model_name in models_to_show:
            model_info = self.models[model_name]
            auth_req = "🔒" if model_info.get('requires_auth', False) else "🔓"
            size = model_info.get('approximate_size_gb', 0)
            
            print(f"{auth_req} {model_name:<20} | {model_info['size']:<15} | {size:.1f} GB | {model_info['description']}")
        
        print("\n📁 Available Collections:")
        for collection_name, models in self.collections.items():
            total_size = sum(self.models[m].get('approximate_size_gb', 0) for m in models)  
            print(f"  {collection_name:<20} | {len(models)} models | ~{total_size:.1f} GB")
        
        print(f"\n🔒 = Requires authentication | 🔓 = No authentication required")
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if model is already downloaded."""
        model_path = self.models_dir / model_name
        
        # Check if directory exists and has model files
        if not model_path.exists():
            return False
        
        # Look for common model files
        model_files = ['pytorch_model.bin', 'model.safetensors', 'config.json']
        has_model_file = any((model_path / f).exists() for f in model_files)
        
        return has_model_file
    
    def download_model(self, model_name: str, force_redownload: bool = False) -> bool:
        """
        Download a single model.
        
        Args:
            model_name: Name of model to download
            force_redownload: Whether to redownload if already exists
            
        Returns:
            True if download successful
        """
        model_info = self.get_model_info(model_name)
        if not model_info:
            print(f"❌ Model '{model_name}' not found in registry")
            return False
        
        # Check if already downloaded
        if self.is_model_downloaded(model_name) and not force_redownload:
            print(f"✅ Model '{model_name}' already downloaded")
            return True
        
        # Check authentication requirement
        if model_info.get('requires_auth', False) and not self.authenticated:
            print(f"❌ Model '{model_name}' requires authentication")
            print(f"Note: {model_info.get('auth_note', 'Please authenticate first')}")
            return False
        
        # Check storage space
        required_gb = model_info.get('approximate_size_gb', 5)  # Default 5GB
        if not self.check_storage_space(required_gb + self.storage_req['minimum_free_space_gb']):
            return False
        
        try:
            print(f"🔽 Downloading {model_name} ({model_info['size']})...")
            
            huggingface_id = model_info['huggingface_id']
            model_path = self.models_dir / model_name
            
            # Download model files
            print(f"  Downloading from: {huggingface_id}")
            
            snapshot_download(
                repo_id=huggingface_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=self.settings.get('use_symlinks', True),
                resume_download=True,
                cache_dir=str(self.cache_dir)
            )
            
            # Verify download by loading config
            try:
                if model_info['model_type'] == 'decoder':
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                    model = AutoModelForCausalLM.from_pretrained(
                        str(model_path), 
                        torch_dtype=torch.float16,
                        device_map="cpu"  # Load on CPU for verification
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                    model = AutoModel.from_pretrained(str(model_path))
                
                print(f"✅ Successfully downloaded and verified: {model_name}")
                print(f"   Location: {model_path}")
                print(f"   Model layers: {len(model.parameters()) if hasattr(model, 'parameters') else 'N/A'}")
                
                # Clean up model from memory
                del model, tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                return True
                
            except Exception as e:
                print(f"⚠️  Downloaded but verification failed: {e}")
                print(f"   Model files are at: {model_path}")
                return True  # Consider successful if files are downloaded
                
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            
            # Clean up partial download
            model_path = self.models_dir / model_name
            if model_path.exists():
                try:
                    shutil.rmtree(model_path)
                    print(f"   Cleaned up partial download")
                except:
                    pass
            
            return False
    
    def download_collection(self, collection_name: str, 
                          force_redownload: bool = False) -> Tuple[int, int]:
        """
        Download all models in a collection.
        
        Args:
            collection_name: Name of collection to download
            force_redownload: Whether to redownload existing models
            
        Returns:
            Tuple of (successful_downloads, total_models)
        """
        if collection_name not in self.collections:
            print(f"❌ Collection '{collection_name}' not found")
            available = ", ".join(self.collections.keys())
            print(f"Available collections: {available}")
            return 0, 0
        
        models = self.collections[collection_name]
        print(f"📦 Downloading collection '{collection_name}' ({len(models)} models)")
        
        successful = 0
        for i, model_name in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}] Processing {model_name}...")
            if self.download_model(model_name, force_redownload):
                successful += 1
            else:
                print(f"⚠️  Skipping {model_name} due to error")
        
        print(f"\n📊 Collection download summary:")
        print(f"   Successfully downloaded: {successful}/{len(models)} models")
        
        return successful, len(models)
    
    def download_multiple(self, model_names: List[str], 
                         force_redownload: bool = False) -> Tuple[int, int]:
        """
        Download multiple models.
        
        Args:
            model_names: List of model names to download
            force_redownload: Whether to redownload existing models
            
        Returns:
            Tuple of (successful_downloads, total_models)
        """
        print(f"📦 Downloading {len(model_names)} models...")
        
        successful = 0
        for i, model_name in enumerate(model_names, 1):
            print(f"\n[{i}/{len(model_names)}] Processing {model_name}...")
            if self.download_model(model_name, force_redownload):
                successful += 1
        
        print(f"\n📊 Download summary:")
        print(f"   Successfully downloaded: {successful}/{len(model_names)} models")
        
        return successful, len(model_names)
    
    def download_all(self, force_redownload: bool = False) -> Tuple[int, int]:
        """Download all models in registry."""
        all_models = list(self.models.keys())
        return self.download_multiple(all_models, force_redownload)
    
    def get_download_status(self) -> Dict[str, Any]:
        """Get status of all model downloads."""
        status = {
            'downloaded': [],
            'not_downloaded': [],
            'total_size_gb': 0,
            'downloaded_size_gb': 0
        }
        
        for model_name, model_info in self.models.items():
            size_gb = model_info.get('approximate_size_gb', 0)
            status['total_size_gb'] += size_gb
            
            if self.is_model_downloaded(model_name):
                status['downloaded'].append(model_name)
                status['downloaded_size_gb'] += size_gb
            else:
                status['not_downloaded'].append(model_name)
        
        return status
    
    def cleanup_failed_downloads(self) -> None:
        """Clean up any failed or partial downloads."""
        print("🧹 Cleaning up failed downloads...")
        
        cleaned = 0
        for model_name in self.models.keys():
            model_path = self.models_dir / model_name
            
            if model_path.exists() and not self.is_model_downloaded(model_name):
                try:
                    shutil.rmtree(model_path)
                    print(f"   Removed incomplete: {model_name}")
                    cleaned += 1
                except Exception as e:
                    print(f"   Failed to remove {model_name}: {e}")
        
        if cleaned == 0:
            print("   No failed downloads found")
        else:
            print(f"   Cleaned up {cleaned} failed downloads")


def main():
    """Main entry point for model download manager."""
    parser = argparse.ArgumentParser(description="Download and manage open-source models")
    parser.add_argument("--models", nargs="+", help="Specific models to download")
    parser.add_argument("--collection", help="Download entire model collection")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--status", action="store_true", help="Show download status")
    parser.add_argument("--cleanup", action="store_true", help="Clean up failed downloads")
    parser.add_argument("--force", action="store_true", help="Force redownload existing models")
    parser.add_argument("--token", help="HuggingFace authentication token")
    parser.add_argument("--models-dir", default="./models", help="Directory to store models")
    
    args = parser.parse_args()
    
    # Initialize manager
    try:
        manager = ModelDownloadManager(models_dir=args.models_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Handle list command
    if args.list:
        manager.list_models()
        return
    
    # Handle status command
    if args.status:
        status = manager.get_download_status()
        print(f"📊 Download Status:")
        print(f"   Downloaded: {len(status['downloaded'])}/{len(manager.models)} models")
        print(f"   Size: {status['downloaded_size_gb']:.1f}/{status['total_size_gb']:.1f} GB")
        print(f"\n✅ Downloaded models: {', '.join(status['downloaded'])}")
        if status['not_downloaded']:
            print(f"⏳ Not downloaded: {', '.join(status['not_downloaded'])}")
        return
    
    # Handle cleanup command  
    if args.cleanup:
        manager.cleanup_failed_downloads()
        return
    
    # Authenticate if token provided
    if args.token:
        manager.authenticate_huggingface(args.token)
    
    # Handle download commands
    if args.all:
        print("🚀 Downloading ALL models...")
        manager.download_all(args.force)
    elif args.collection:
        manager.download_collection(args.collection, args.force)
    elif args.models:
        manager.download_multiple(args.models, args.force)
    else:
        print("No download action specified. Use --help for usage information.")
        manager.list_models()


if __name__ == "__main__":
    main()