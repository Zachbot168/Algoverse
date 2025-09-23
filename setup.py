#!/usr/bin/env python3
"""
Algoverse Setup Script
Automated setup for the FIRM bias mitigation framework
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} compatible")
    return True

def check_cuda():
    """Check CUDA availability."""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA detected")
            return True
    except:
        pass
    print("⚠️  CUDA not detected - will use CPU mode")
    return False

def setup_environment():
    """Set up environment variables."""
    env_vars = {
        'TORCH_DYNAMO_DISABLE': '1',
        'TORCH_COMPILE_DEBUG': '0',
        'TOKENIZERS_PARALLELISM': 'false',
        'TRANSFORMERS_VERBOSITY': 'error',
        'TRANSFORMERS_NO_ADVISORY_WARNINGS': 'true',
        'HF_HUB_DISABLE_PROGRESS_BARS': 'true',
        'PYTHONWARNINGS': 'ignore'
    }
    
    env_content = "# Algoverse Environment Variables\n"
    for key, value in env_vars.items():
        env_content += f"export {key}={value}\n"
        os.environ[key] = value
    
    # Write to .env file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Environment variables configured")

def main():
    """Main setup function."""
    print("🚀 Algoverse FIRM Framework Setup")
    print("=" * 40)
    
    # Check requirements
    if not check_python_version():
        sys.exit(1)
    
    has_cuda = check_cuda()
    
    # Setup environment
    setup_environment()
    
    # Install PyTorch
    if has_cuda:
        torch_cmd = "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118"
    else:
        torch_cmd = "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu"
    
    if not run_command(torch_cmd, "Installing PyTorch"):
        print("⚠️  PyTorch installation failed - continuing with other dependencies")
    
    # Install requirements
    run_command("pip install -r requirements.txt", "Installing Python dependencies")
    
    # Install spaCy model
    run_command("python -m spacy download en_core_web_sm", "Installing spaCy English model")
    
    # Make scripts executable
    run_command("chmod +x enhanced_pull_datasets.sh", "Making dataset script executable")
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Run dataset download: ./enhanced_pull_datasets.sh")
    print("2. Authenticate with HuggingFace: huggingface-cli login")
    print("3. Test installation: cd unified_pipeline && python test_new_models.py")
    print("4. Run quick evaluation: python run_unified_pipeline.py --model-config configs/models/gemma-2-2b-it.yaml --suite quick_evaluation")

if __name__ == "__main__":
    main()